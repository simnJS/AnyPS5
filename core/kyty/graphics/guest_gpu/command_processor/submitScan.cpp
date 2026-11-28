#include "graphics/guest_gpu/command_processor/submitScan.h"

#include "common/fallback.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/renderContext.h"


namespace Libs::Graphics {

namespace {

constexpr uint32_t MaxDepth = 4;
constexpr uint64_t BudgetDw = uint64_t {1} << 22u;

class Scan {
public:
	Scan(CommandProcessor& shadow, const RenderContext& renderer, SubmitScanSink& sink)
	    : m_shadow(shadow), m_renderer(renderer), m_sink(sink) {}

	void Walk(std::span<const uint32_t> commands, uint32_t depth) {
		for (size_t position = 0; position < commands.size() && Running();) {
			const uint32_t header = commands[position];
			if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
				position++;
				continue;
			}
			if ((header >> 30u) != 3u) {
				Stop(SubmitScanEnd::Malformed);
				return;
			}
			const uint32_t length = KYTY_PM4_LEN(header);
			if (length > commands.size() - position) {
				Stop(SubmitScanEnd::Malformed);
				return;
			}
			if (length > m_budget) {
				Stop(SubmitScanEnd::Budget);
				return;
			}
			m_budget -= length;
			const auto     body   = commands.subspan(position + 1, length - 1);
			const uint32_t opcode = (header >> 8u) & 0xffu;
			position += length;
			if (opcode == Pm4::IT_REWIND) {
				Stop(SubmitScanEnd::Rewind);
				return;
			}
			if (opcode != Pm4::IT_INDIRECT_BUFFER) {
				Packet(header, body);
				continue;
			}
			if (body.size() == 13) {
				if ((body[0] & 0x3u) == 2u && depth < MaxDepth) {
					Walk(Buffer(body.subspan(10)), depth + 1);
				}
				commands = Buffer(body.subspan(7));
				position = 0;
				continue;
			}
			if (body.size() != 3) {
				Stop(SubmitScanEnd::Malformed);
				return;
			}
			if ((body[2] & (1u << 20u)) != 0) {
				commands = Buffer(body);
				position = 0;
			} else if (depth < MaxDepth) {
				Walk(Buffer(body), depth + 1);
			}
		}
	}

	[[nodiscard]] SubmitScanResult Result() const { return m_result; }
	void                           Stop(SubmitScanEnd end) {
		if (m_result.end == SubmitScanEnd::Complete) m_result.end = end;
	}

private:
	[[nodiscard]] bool Running() const { return m_result.end == SubmitScanEnd::Complete; }

	[[nodiscard]] std::span<const uint32_t> Buffer(std::span<const uint32_t> body) {
		const uint64_t address = (body[0] & ~0x3u) | (static_cast<uint64_t>(body[1]) << 32u);
		const uint32_t dwords  = body[2] & 0xfffffu;
		if (address == 0 || dwords == 0) return {};
		if (!m_renderer.IsMapped(address, uint64_t {dwords} * 4u)) {
			Stop(SubmitScanEnd::Unreadable);
			return {};
		}
		return {reinterpret_cast<const uint32_t*>(address), dwords};
	}

	[[nodiscard]] bool IndirectRegistersReadable(std::span<const uint32_t> body) const {
		if (body.size() != 4) return false;
		const uint64_t address = (body[0] & 0xfffffffcu) | (static_cast<uint64_t>(body[1]) << 32u);
		const uint32_t count   = body[3] & 0x3fffu;
		return count == 0 || m_renderer.IsMapped(address, uint64_t {count} * 8u);
	}

	void Apply(uint32_t header, std::span<const uint32_t> body, cp_op_parser_func_t handler) {
		const auto size = static_cast<uint32_t>(body.size() + 1);
		(void)handler(m_shadow, header & ~1u, body.data(), size, size);
	}

	void Packet(uint32_t header, std::span<const uint32_t> body) {
		const uint32_t opcode = (header >> 8u) & 0xffu;
		if (body.empty()) return;
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_CLEAR_STATE:
			case Pm4::IT_SET_BASE: Apply(header, body, g_cp_op_func[opcode]); break;
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
				if (!IndirectRegistersReadable(body)) {
					Stop(SubmitScanEnd::Unreadable);
					return;
				}
				Apply(header, body, g_cp_op_func[opcode]);
				break;
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX:
				if (body[0] == Pm4::SQ_THREAD_TRACE_USERDATA_2) {
					if ((KYTY_PM4_R(header) == 1u || m_shadow.IsAsyncComputeQueue()) && body.size() >= 2 &&
					    (body[1] & 0xff000000u) == Pm4::USERDATA_TAG_RESET_QUEUE) {
						m_shadow.ResetQueue(body[1]);
					}
				} else if (KYTY_PM4_R(header) == 0u) {
					Apply(header, body, g_cp_op_func[opcode]);
				}
				break;
			case Pm4::IT_NOP:
				if (KYTY_PM4_R(header) == Pm4::R_CONTEXT_STATE) {
					Apply(header, body, g_cp_op_custom_func[Pm4::R_CONTEXT_STATE]);
				}
				break;
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_MULTI_INST:
			case Pm4::IT_DRAW_INDEX_AUTO:
				m_result.draws++;
				m_sink.Draw(m_shadow, nullptr);
				break;
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
				if (body.size() >= 4 && (header == 0xc0032400u || header == 0xc0032500u)) {
					const bool                  indexed = header == 0xc0032500u;
					const IndirectDrawRegisters registers {
					    body[1] & 0xffffu, body[2] & 0xffffu,
					    indexed && (body[2] & (1u << 28u)) != 0 ? body[1] >> 16u : Pm4::SH_NOP, Pm4::SH_NOP};
					m_result.draws++;
					m_sink.Draw(m_shadow, &registers);
				}
				break;
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
				if (body.size() >= 9 && (header == 0xc0082c00u || header == 0xc0083800u)) {
					const bool                  indexed = header == 0xc0083800u;
					const IndirectDrawRegisters registers {
					    body[1] & 0xffffu, body[2] & 0xffffu,
					    indexed && (body[3] & (1u << 28u)) != 0 ? body[1] >> 16u : Pm4::SH_NOP,
					    (body[3] >> 31u) != 0 ? body[3] & 0xffffu : Pm4::SH_NOP};
					m_result.draws++;
					m_sink.Draw(m_shadow, &registers);
				}
				break;
			case Pm4::IT_DISPATCH_DIRECT:
				if (body.size() == 4) {
					SubmitScanDispatch dispatch {.thread_groups = {body[0], body[1], body[2]}, .mode = body[3]};
					Dispatch(dispatch);
				}
				break;
			case Pm4::IT_DISPATCH_INDIRECT:
				if (header == 0xc0021600u && body.size() == 3) {
					Dispatch({.mode      = body[2],
					          .args_addr = body[0] | (static_cast<uint64_t>(body[1]) << 32u),
					          .indirect  = true});
				} else if (header == 0xc0011600u && body.size() == 2) {
					const auto base = m_shadow.GetDispatchIndirectArgsBaseAddress();
					if (base != 0) Dispatch({.mode = body[1], .args_addr = base + body[0], .indirect = true});
				}
				break;
			default: break;
		}
	}

	void Dispatch(const SubmitScanDispatch& dispatch) {
		if (dispatch.indirect && (dispatch.args_addr == 0 || (dispatch.args_addr & 3u) != 0)) return;
		m_shadow.GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(dispatch.mode));
		m_result.dispatches++;
		m_sink.Dispatch(m_shadow, dispatch);
	}

	CommandProcessor&    m_shadow;
	const RenderContext& m_renderer;
	SubmitScanSink&      m_sink;
	SubmitScanResult     m_result;
	uint64_t             m_budget = BudgetDw;
};

}

SubmitScanResult ScanSubmission(std::span<const uint32_t> commands, CommandProcessor& shadow,
                                const RenderContext& renderer, SubmitScanSink& sink) {
	Scan scan(shadow, renderer, sink);
	if (!Common::Fallback::Run("submission scan stopped: the shadow command processor refused a packet",
	                           "ScanSubmission", 0, 0, true, [&] { scan.Walk(commands, 0); })) {
		scan.Stop(SubmitScanEnd::Error);
		shadow.Reset();
	}
	return scan.Result();
}

}
