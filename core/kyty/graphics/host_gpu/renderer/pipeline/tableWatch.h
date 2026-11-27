#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TABLEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TABLEWATCH_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/image/imageChangeJournal.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

using TableElementJournal = ChangeJournal<16>;

class TableWatch final {
public:
	static constexpr uint32_t None = UINT32_MAX;

	[[nodiscard]] bool Valid() const noexcept { return m_valid; }
	void               Invalidate() noexcept { m_valid = false; }

	void Begin(uint32_t count) {
		m_next.resize(count, None);
		m_watched.resize(count, None);
		m_considered.resize(count, 0);
		m_count = count;
		m_stamp++;
	}

	[[nodiscard]] bool Consider(uint32_t element) {
		if (element >= m_count || m_considered[element] == m_stamp) {
			return false;
		}
		m_considered[element] = m_stamp;
		return true;
	}

	template <typename Fn>
	[[nodiscard]] bool Collect(const ImageChangeJournal& images, uint64_t image_head,
	                           const TableElementJournal& elements, uint64_t element_head,
	                           std::span<const uint32_t> bound_images, Fn&& visit) {
		if (!m_valid) {
			return false;
		}
		for (const auto element: m_pending) {
			visit(element);
		}
		const auto visit_image = [&](uint32_t image) {
			const auto head = m_heads.find(image);
			for (auto element = head == m_heads.end() ? None : head->second; element != None;
			     element = m_next[element]) {
				visit(element);
			}
		};
		if (!images.ForEach(m_image_seq, image_head, visit_image)) {
			return false;
		}
		const bool known = elements.ForEach(m_element_seq, element_head, [&](uint32_t slot) {
			if (const auto found = m_slots.find(slot); found != m_slots.end()) {
				visit(found->second);
			}
		});
		if (!known) {
			return false;
		}
		for (const auto image: bound_images) {
			visit_image(image);
		}
		return true;
	}

	void BeginTracking(bool rebuild) {
		m_pending.clear();
		m_conflict = false;
		if (rebuild) {
			m_heads.clear();
			m_slots.clear();
			m_watched.assign(m_count, None);
			m_next.assign(m_count, None);
		}
	}

	template <typename Claims>
	void Track(uint32_t element, bool checked, bool has_image, uint32_t image, uint32_t slot, Claims&& claims) {
		if (!checked) {
			m_pending.push_back(element);
			return;
		}
		if (has_image) {
			Watch(element, image);
		} else {
			Unwatch(element);
		}
		const auto [found, inserted] = m_slots.try_emplace(slot, element);
		if (inserted || found->second == element) {
			return;
		}
		if (claims(found->second, slot)) {
			m_conflict = true;
			return;
		}
		found->second = element;
	}

	void Finish(uint64_t image_head, uint64_t element_head) {
		m_image_seq   = image_head;
		m_element_seq = element_head;
		m_valid       = !m_conflict;
	}

	[[nodiscard]] size_t Pending() const noexcept { return m_pending.size(); }

private:
	void Unwatch(uint32_t element) {
		const auto image = m_watched[element];
		if (image == None) {
			return;
		}
		const auto head = m_heads.find(image);
		EXIT_IF(head == m_heads.end());
		if (head->second == element) {
			if (m_next[element] == None) {
				m_heads.erase(head);
			} else {
				head->second = m_next[element];
			}
		} else {
			auto at = head->second;
			while (m_next[at] != element) {
				at = m_next[at];
				EXIT_IF(at == None);
			}
			m_next[at] = m_next[element];
		}
		m_next[element]    = None;
		m_watched[element] = None;
	}

	void Watch(uint32_t element, uint32_t image) {
		if (m_watched[element] == image) {
			return;
		}
		Unwatch(element);
		const auto [head, inserted] = m_heads.try_emplace(image, element);
		m_next[element]             = inserted ? None : head->second;
		head->second                = element;
		m_watched[element]          = image;
	}

	bool                                   m_valid       = false;
	bool                                   m_conflict    = false;
	uint32_t                               m_count       = 0;
	uint64_t                               m_image_seq   = 0;
	uint64_t                               m_element_seq = 0;
	uint64_t                               m_stamp       = 0;
	std::unordered_map<uint32_t, uint32_t> m_heads;
	std::unordered_map<uint32_t, uint32_t> m_slots;
	std::vector<uint32_t>                  m_next;
	std::vector<uint32_t>                  m_watched;
	std::vector<uint64_t>                  m_considered;
	std::vector<uint32_t>                  m_pending;
};

}

#endif
