import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "stall-report.py"
spec = importlib.util.spec_from_file_location("stall_report", TOOL)
tool = importlib.util.module_from_spec(spec)
sys.modules["stall_report"] = tool
spec.loader.exec_module(tool)

PIXEL = "ud=6 cur=19 wave=32 smask=1f"
MESH = "ud=20 cur=33 idraw"
COMPUTE = "ud=8 t=32,1,1 wave=32"
STATE = "cc=1 c0=R8G8B8A8Unorm depth=D32Sfloat m0=f"


class StallReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "_PipelineCache").mkdir()

    def session(self, log, prepared, recorded, known, loaded=None, store=True):
        cache = self.root / "_PipelineCache"
        modules = sum(line.startswith("M ") for line in prepared) if loaded is None else loaded[0]
        pipelines = sum(line.startswith("P ") for line in prepared) if loaded is None else loaded[1]
        header = [
            "[kyty-startup] preparing pipelines before guest entry",
            f"Pipeline manifest: {modules} modules, {pipelines} pipelines in _PipelineCache/title.shared.kpm (loaded in 1.0 ms)",
            f"Pipeline manifest: the code of 1 shaders comes from _PipelineCache/title.code ({len(known)} shaders)",
            "[kyty-startup] guest_entry monotonic_ns=1000000000",
        ]
        (self.root / "stderr.log").write_text("\n".join(header + log) + "\n", encoding="utf-8")
        (cache / "title.shared.kpm").write_text("# manifest\n" + "\n".join(prepared) + "\n", encoding="utf-8")
        (cache / "title.kpm").write_text("# manifest\n" + "\n".join(recorded) + "\n", encoding="utf-8")
        if store:
            with (cache / "title.code").open("wb") as stream:
                stream.write(struct.pack("<Q", tool.CODE_STORE_MAGIC))
                for program in known:
                    stream.write(struct.pack("<QII", int(program, 16), 1, 1) + b"\0")
        return tool.Session(*tool.find_inputs(self.root))

    def causes(self, session):
        return [(wait.kind, wait.program, wait.cause) for wait in session.waits]

    def test_unknown_refused_and_unprepared_programs(self):
        known = ["00000000000000a1", "00000000000000a2"]
        session = self.session(
            [
                "[kyty-startup] generic_refused stage=4 program=00000000000000a1 variant=1111111111111111 reason=several_indirect_tables",
                "[kyty-gpl] stall 1.000 s graphics-sync 0x00000000000000a1 linked fs=queued pr=queued why=eligible @2000000000",
                "[kyty-gpl] stall 0.500 s compute-sync 0x00000000000000a2  why=writes-buffers @3000000000",
                "[kyty-gpl] translate 0.250 s 0x00000000000000a3 @3500000000",
                "[kyty-gpl] stall 2.000 s compute-sync 0x00000000000000a3  why=writes-images @6000000000",
            ],
            [f"M 000000000001 ps code=00000000000000a1 {PIXEL} sb=st:48"], [], known)
        self.assertEqual(self.causes(session), [
            ("graphics-sync", "00000000000000a1", "generic-refused:several_indirect_tables"),
            ("compute-sync", "00000000000000a2", "generic-not-prepared:no-recipe"),
            ("translate", "00000000000000a3", "unknown-program"),
            ("compute-sync", "00000000000000a3", "unknown-program"),
        ])
        summary = session.summary()
        self.assertEqual((summary["known_waits"], summary["known_seconds"]), (2, 1.5))
        self.assertEqual((summary["unknown_waits"], summary["unknown_seconds"]), (2, 2.25))
        self.assertEqual(summary["families"]["generic-refused"]["seconds"], 1.0)

    def test_bounds_bits_on_the_pixel_or_the_paired_stage(self):
        prepared = [
            f"M 000000000001 ps code=00000000000000b1 {PIXEL} sb=st:48,st:1",
            f"M 000000000002 ps code=00000000000000b1 {PIXEL} gg",
            f"M 000000000003 mesh code=00000000000000b2 {MESH} sb=st:4",
            f"M 000000000004 mesh code=00000000000000b2 {MESH} gg",
            f"M 000000000005 ps code=00000000000000b3 {PIXEL} sb=st:32",
            f"P v=000000000003 p=000000000001 {STATE}",
            f"P v=000000000003 p=000000000005 {STATE}",
        ]
        recorded = prepared + [
            f"M 000000000011 ps code=00000000000000b1 {PIXEL} sb=st:2097200,st:4194305",
            f"M 000000000013 mesh code=00000000000000b2 {MESH} sb=st:2097156",
            f"P v=000000000003 p=000000000011 {STATE}",
            f"P v=000000000013 p=000000000005 {STATE}",
        ]
        session = self.session(
            [
                "[kyty-gpl] translate 0.108 s 0x00000000000000b1 @1900000000",
                "[kyty-gpl] stall 1.250 s graphics-sync 0x00000000000000b1 linked fs=queued pr=ready why=eligible @3200000000",
                "[kyty-gpl] stall 0.900 s graphics-sync 0x00000000000000b3 linked fs=queued pr=queued why=eligible @4200000000",
                "Pipeline manifest: wrote _PipelineCache/title.kpm (5 modules)",
            ],
            prepared, recorded, ["00000000000000b1", "00000000000000b2", "00000000000000b3"])
        self.assertEqual({wait.cause for wait in session.waits}, {"generic-unbound:buffer-bounds-bits"})
        self.assertEqual([wait.stage for wait in session.waits], ["ps", "ps", "ps"])
        self.assertEqual(session.waits[1].variant, "000000000011")
        self.assertIn("mesh 00000000000000b2", session.waits[2].evidence)
        self.assertEqual(session.summary()["new_variants"], 2)

    def test_render_state_without_a_prepared_pipeline(self):
        prepared = [
            f"M 000000000001 ps code=00000000000000c1 {PIXEL} sb=st:48",
            f"M 000000000003 vs code=00000000000000c2 ud=4 cur=33 sb=st:4",
            f"M 000000000004 ps code=00000000000000c3 {PIXEL} sb=st:48",
            f"P v=000000000003 p=000000000001 {STATE} vb=16.0 va=0.0",
        ]
        recorded = prepared + [
            f"P v=000000000003 p=000000000001 cc=1 c0=R16G16B16A16Sfloat depth=D32Sfloat m0=f b0=1.4.0.5 ms=4 vb=16.0 va=0.0",
        ]
        session = self.session(
            [
                "[kyty-gpl] stall 0.700 s graphics-sync 0x00000000000000c1 linked fs=queued pr=ready why=eligible @3000000000",
                "Pipeline manifest: wrote _PipelineCache/title.kpm (2 modules)",
                "[kyty-gpl] stall 0.300 s graphics-sync 0x00000000000000c3 linked fs=ready pr=ready why=eligible @5000000000",
            ],
            prepared, recorded, ["00000000000000c1", "00000000000000c2", "00000000000000c3"])
        self.assertEqual(session.waits[0].cause, "library-missing:fs+fo")
        self.assertIn("b0, c0, ms", session.waits[0].evidence)
        self.assertEqual(session.waits[1].cause, "other:not-recorded")

    def test_compute_keys_recipes_and_refusals(self):
        images = ",".join(["n:Float/d:2D"] * 4)
        prepared = [
            f"M 000000000001 cs code=00000000000000d1 {COMPUTE} rs sb=st:16383 si={images}",
            f"M 000000000002 cs code=00000000000000d2 {COMPUTE} sb=st:16383,st:16383 si={images}",
            f"M 000000000003 cs code=00000000000000d3 {COMPUTE} sb=st:2113535 si=n:Float/d:2D/t:0/tc:65536",
            f"M 000000000004 cs code=00000000000000d4 {COMPUTE} sb=st:4210687",
        ]
        recorded = [line.replace("st:2113535", "st:16383").replace("st:4210687", "st:16383") for line in prepared] + [
            f"M 000000000011 cs code=00000000000000d1 {COMPUTE} sb=st:16383 si={images}",
            f"M 000000000012 cs code=00000000000000d2 {COMPUTE} sb=st:16383,st:16383,st:16383 si={images},n:Uint/d:2D",
            f"M 000000000013 cs code=00000000000000d3 {COMPUTE} sb=st:16383 si=n:Float/d:2D/t:0/tc:32768",
            f"M 000000000014 cs code=00000000000000d4 {COMPUTE} sb=st:16383",
        ]
        session = self.session(
            [
                "[kyty-startup] unresolved_module stage=3 program=00000000000000d2 variant=2222222222222222 generic=0 reason=failed detail=specialization_does_not_fit",
                "[kyty-gpl] stall 12.000 s compute-sync 0x00000000000000d1  why=writes-buffers @20000000000",
                "[kyty-gpl] stall 3.500 s compute-sync 0x00000000000000d2  why=writes-images @30000000000",
                "[kyty-gpl] stall 9.000 s compute-sync 0x00000000000000d3  why=writes-images @40000000000",
                "[kyty-gpl] stall 0.400 s compute-wait 0x00000000000000d4  why=writes-nothing @50000000000",
                "[kyty-gpl] stall 0.300 s compute-wait 0x00000000000000d4  why=writes-nothing @69000000000",
                "[kyty-gpl] stall 0.200 s compute-sync 0x00000000000000d1  why=writes-buffers @70000000000",
                "Pipeline manifest: wrote _PipelineCache/title.kpm (8 modules)",
                "[kyty-gpl] stall 0.100 s compute-wait 0x00000000000000d4  why=writes-nothing @90000000000",
            ],
            prepared, recorded, [f"00000000000000d{index}" for index in range(1, 5)])
        self.assertEqual([wait.cause for wait in session.waits], [
            "other-key:rs",
            "generic-refused:specialization_does_not_fit",
            "generic-unbound:image-table-capacity(65536->32768)",
            "other:prepared-module-without-driver-pipeline",
            "other:not-recorded",
            "other-key:rs",
            "other:not-recorded",
        ])
        self.assertIn("less than 2 s before the last write", session.waits[4].evidence)
        self.assertIn("last written before this wait", session.waits[6].evidence)
        self.assertTrue(session.canonical_strides)

    def test_nested_translations_and_a_program_prepared_under_the_run_time_key_only(self):
        prepared = [
            f"M 000000000001 cs code=00000000000000e7 {COMPUTE} rs sb=st:16383 si=n:Float/d:2D ss=ir:0/im:4/is:5",
            f"M 000000000002 ps code=00000000000000e8 {PIXEL} sb=st:48",
            f"M 000000000003 vs code=00000000000000e9 ud=4 cur=33",
            f"P v=000000000003 p=000000000002 {STATE}",
        ]
        session = self.session(
            [
                "[kyty] shader 0x00000000000000e7: its sampler descriptors chosen in memory tables are read at run time and "
                "served from a fixed set of samplers when its candidates cannot be bound",
                "[kyty-gpl] translate 0.534 s 0x00000000000000e7 @5000000000",
                "[kyty-gpl] translate 1.247 s 0x00000000000000e7 @5000030000",
                "[kyty-gpl] translate 0.200 s 0x00000000000000e7 cause=generic-unbound:image-heap variant=00000000000000aa "
                "@8000000000",
                "[kyty-gpl] translate 0.900 s 0x00000000000000e7 @8000020000",
                "[kyty-gpl] stall 0.300 s graphics-sync 0x00000000000000e8 linked fs=queued pr=ready why=eligible @20000000000",
                "[kyty-gpl] stall 0.100 s graphics-sync 0x00000000000000e8 linked fs=ready pr=ready why=eligible @21000000000",
                "[kyty-gpl] stall 0.100 s compute-sync 0x00000000000000e7  why=writes-buffers @60000000000",
                "Pipeline manifest: wrote _PipelineCache/title.kpm (3 modules)",
            ],
            prepared, prepared, ["00000000000000e7", "00000000000000e8", "00000000000000e9"])
        self.assertEqual([(wait.kind, wait.seconds, wait.cause) for wait in session.waits[:4]], [
            ("translate", 1.247, "other-key:rs"),
            ("translate", 0.9, "generic-unbound:image-heap"),
            ("graphics-sync", 0.3, "library-missing:fs"),
            ("graphics-sync", 0.1, "library-missing:vi+fo"),
        ])
        self.assertEqual((session.waits[1].variant, session.waits[1].engine_cause), ("00000000000000aa", True))
        summary = session.summary()
        self.assertEqual((summary["nested_translations"], summary["waits"]), (2, 5))
        self.assertEqual(summary["seconds"], 2.647)

    def test_refusal_named_by_the_compute_service(self):
        prepared = [f"M 000000000001 cs code=00000000000000e5 {COMPUTE} sb=st:16383 si=n:Float/d:2D/heap"]
        recorded = prepared + [f"M 000000000011 cs code=00000000000000e5 {COMPUTE} sb=st:16383 si=n:Float/d:2D/t:0/tc:32768"]
        session = self.session(
            [
                "[kyty-gpl] unserved compute 0x00000000000000e5 variants 1 closest image-addressing slot 37 after-rebind @5000000000",
                "[kyty-gpl] translate 0.300 s 0x00000000000000e5 @5400000000",
                "[kyty-gpl] stall 4.000 s compute-sync 0x00000000000000e5  why=writes-images @9500000000",
                "[kyty-gpl] stall 1.000 s compute-sync 0x00000000000000e5  why=writes-images @60000000000",
                "Pipeline manifest: wrote _PipelineCache/title.kpm (2 modules)",
            ],
            prepared, recorded, ["00000000000000e5"])
        self.assertEqual([wait.cause for wait in session.waits], [
            "generic-unbound:image-addressing:37:after-rebind",
            "generic-unbound:image-addressing:37:after-rebind",
            "generic-unbound:image-heap+image-t+image-table-capacity(0->32768)",
        ])

    def test_bounds_bits_kept_in_the_key_by_an_older_engine(self):
        prepared = [f"M 000000000001 cs code=00000000000000e1 {COMPUTE} sb=st:16383"]
        recorded = prepared + [f"M 000000000011 cs code=00000000000000e1 {COMPUTE} sb=st:2113535"]
        session = self.session(
            ["[kyty-gpl] stall 4.000 s compute-sync 0x00000000000000e1  why=writes-buffers @9000000000"],
            prepared, recorded, ["00000000000000e1"])
        self.assertFalse(session.canonical_strides)
        self.assertEqual(session.waits[0].cause, "generic-unbound:buffer-bounds-bits")

    def test_engine_cause_shared_limit_and_unfinished_wait(self):
        prepared = [
            f"M 000000000001 ps code=00000000000000f1 {PIXEL} sb=st:48",
            f"M 000000000002 ps code=00000000000000f2 {PIXEL} sb=st:48",
        ]
        session = self.session(
            [
                "[kyty-gpl] stall 1.500 s graphics-sync 0x00000000000000f1 linked fs=queued pr=queued why=eligible "
                "cause=generic-unbound:image-format variant=0123456789abcdef @4000000000",
                "[kyty-gpl] stall 0.800 s graphics-sync 0x00000000000000f2 linked fs=queued pr=queued why=eligible "
                "cause=unknown-program @5000000000",
                "[kyty-gpl] waiting_shader shader=00000000000000f3 elapsed_s=3.1 monotonic_ns=8100000000",
                "[kyty-gpl] stall 4.200 s compute-sync 0x00000000000000f3  why=writes-buffers @9200000000",
                "[kyty-gpl] waiting_shader shader=00000000000000f1 elapsed_s=3.0 monotonic_ns=20000000000",
            ],
            prepared, [], ["00000000000000f1", "00000000000000f3"], loaded=(1, 0))
        self.assertEqual(len(session.prepared.modules), 1)
        first, second, third, last = session.waits
        self.assertEqual((first.cause, first.variant, first.engine_cause, first.route),
                         ("generic-unbound:image-format", "0123456789abcdef", True, "linked fs=queued pr=queued"))
        self.assertEqual((second.cause, second.known), ("unknown-program", False))
        self.assertTrue(third.announced)
        self.assertEqual((last.kind, last.program, last.seconds), ("unfinished", "00000000000000f1", 3.0))
        summary = session.summary()
        self.assertEqual((summary["waits"], summary["engine_causes"], summary["unknown_waits"]), (4, 2, 1))
        self.assertEqual(session.timeline(5), [
            {"from_s": 0, "waits": 2, "seconds": 2.3, "main_cause": "generic-unbound:image-format", "main_seconds": 1.5},
            {"from_s": 5, "waits": 1, "seconds": 4.2, "main_cause": "generic-not-prepared:no-recipe", "main_seconds": 4.2},
            {"from_s": 15, "waits": 1, "seconds": 3.0, "main_cause": "other:prepared-module-without-driver-pipeline",
             "main_seconds": 3.0},
        ])

    def test_variant_map_ranks_the_fields_that_make_variants(self):
        search = "n:Float/d:2D/ir:0"
        prepared = [
            f"M 000000000001 cs code=0000000000000ab1 {COMPUTE} sb=st:16,st:1 si=n:Float/d:2D/heap",
            f"M 000000000002 cs code=0000000000000ab1 {COMPUTE} sb=st:2097168,st:4194305 si=n:Float/d:2D/heap",
            f"M 000000000003 cs code=0000000000000ab1 {COMPUTE} sb=st:16,st:1 si=n:Float/d:2D/t:0/tc:65536",
            f"M 000000000004 cs code=0000000000000ab1 {COMPUTE} async sb=st:16,st:1 si=n:Float/d:2D/t:0/tc:65536",
            f"M 000000000005 ps code=0000000000000ab2 {PIXEL} sb=st:48",
            f"M 000000000006 ps code=0000000000000ab2 {PIXEL} gg",
            f"M 000000000007 vs code=0000000000000ab3 ud=4 cur=33",
            f"P v=000000000007 p=000000000005 {STATE}",
            f"P v=000000000007 p=000000000005 cc=1 c0=R16G16B16A16Sfloat depth=D32Sfloat m0=f",
        ]
        recorded = [line for line in prepared if "st:2097168" not in line] + [
            f"M 000000000011 cs code=0000000000000ab1 {COMPUTE} sb=st:16,st:1 "
            f"si=n:Float/d:2D/ir:0/im:9,{search},{search},n:Uint/d:2D/ir:0",
            f"M 000000000012 ps code=0000000000000ab2 {PIXEL} sb=st:2097200",
            f"M 000000000013 cs code=0000000000000ab4 {COMPUTE} fn=be8a0381be802002",
        ]
        session = self.session([], prepared, recorded, ["0000000000000ab1"])
        report = session.variant_map()
        self.assertEqual((report["programs"], report["loaded"], report["new"], report["programs_with_new"]), (4, 6, 3, 3))
        groups = report["field_groups"]
        self.assertEqual(groups["image-addressing"], {"groups": 1, "values": 2, "merged": 2})
        self.assertEqual(groups["buffer-bounds-bits"], {"groups": 2, "values": 2, "merged": 2})
        self.assertEqual(groups["program-key"], {"groups": 1, "values": 1, "merged": 1})
        self.assertEqual(report["fields"]["key-async-queue"]["merged"], 1)
        self.assertEqual(report["fields"]["image-search-candidates"]["values"], 1)
        self.assertEqual(report["new_fields"], {
            "image-addressing": {"variants": 1, "alone": 1, "programs": 1},
            "buffer-bounds-bits": {"variants": 1, "alone": 1, "programs": 1},
            "no-loaded-recipe": {"variants": 1, "alone": 1, "programs": 1},
        })
        first = report["top_programs"][0]
        self.assertEqual((first["program"], first["loaded"], first["new"]), ("0000000000000ab1", 4, 1))
        self.assertEqual((report["stage_sets"], report["render_states"]), (1, 2))
        self.assertEqual(report["state_fields"]["color-format"], {"groups": 1, "values": 1, "merged": 1})
        text = subprocess.run([sys.executable, "-B", str(TOOL), str(self.root), "--variants"],
                              check=True, capture_output=True, text=True).stdout
        self.assertIn("fields that made the new variants of this session", text)
        self.assertIn("image-addressing", text)

    def test_session_that_never_started_the_title(self):
        cache = self.root / "_PipelineCache"
        (self.root / "stderr.log").write_text(
            "[kyty-startup] preparing pipelines before guest entry\n"
            "Pipeline manifest: 2 modules, 1 pipelines in _PipelineCache/title.shared.kpm (loaded in 1.0 ms)\n"
            "Pipeline manifest: 6000/6675 modules, 167/2559 pipelines (0 modules wait for their shaders) at 2.0 s\n"
            "Pipeline manifest: 6178/6675 modules, 1134/2559 pipelines (0 modules wait for their shaders) at 863.1 s\n",
            encoding="utf-8")
        (cache / "title.shared.kpm").write_text("# manifest\n", encoding="utf-8")
        session = tool.Session(*tool.find_inputs(self.root))
        summary = session.summary()
        self.assertFalse(summary["entered_game"])
        self.assertEqual(summary["startup_progress"], {"modules": 6178, "modules_total": 6675, "pipelines": 1134,
                                                       "pipelines_total": 2559, "seconds": 863.1})
        text = subprocess.run([sys.executable, "-B", str(TOOL), str(self.root)], check=True, capture_output=True,
                              text=True).stdout
        self.assertIn("the title never started in this session (no guest entry in the log); its startup stood at "
                      "6178/6675 modules and 1134/2559 pipelines after 863 s", text)

    def test_command_line_report(self):
        self.session(
            ["[kyty-gpl] stall 0.500 s compute-sync 0x0000000000000aa1  why=writes-buffers @3000000000"],
            [], [], ["0000000000000aa1"])
        text = subprocess.run([sys.executable, "-B", str(TOOL), str(self.root), "--detail"],
                              check=True, capture_output=True, text=True).stdout
        self.assertIn("waits on known programs: 1, 0.5 s", text)
        self.assertNotIn("never started", text)
        self.assertIn("waits on unknown programs: 0, 0.0 s", text)
        self.assertIn("cause=generic-not-prepared:no-recipe", text)
        report = json.loads(subprocess.run([sys.executable, "-B", str(TOOL), str(self.root), "--json", "--detail"],
                                           check=True, capture_output=True, text=True).stdout)
        self.assertEqual(report[0]["list"][0]["program"], "0000000000000aa1")
        self.assertEqual(report[0]["families"]["generic-not-prepared"]["waits"], 1)


if __name__ == "__main__":
    unittest.main()
