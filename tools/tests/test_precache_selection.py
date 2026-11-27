import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PRECACHE = Path(__file__).resolve().parents[1] / 'precache' / 'precache'


class PrecacheSelectionTest(unittest.TestCase):
    def test_generic_draws_preserve_render_signatures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / 'fixture_1.code'
            store.write_bytes(struct.pack('<Q', 0x3145444F4359544B) +
                              b''.join(struct.pack('<QII', code, 1, 1) + b'x' for code in (1, 2, 3)))
            manifest = root / 'input.kpm'
            manifest.write_text('# kyty pipeline manifest v1\n'
                                'M a vs code=0000000000000001 cur=6 wave=32 clip=1 sb=-\n'
                                'M b vs code=0000000000000001 cur=17 clip=1 wave=32 sb=st:4\n'
                                'M c ps code=0000000000000002 cur=12 ss=-\n'
                                'M d cs code=0000000000000003\n'
                                'P v=a p=c c0=R8Unorm\nP c0=R8Unorm p=c v=b\n'
                                'P v=b p=c c0=R8Uint plain\n')
            output = root / 'prepared'
            command = [sys.executable, str(PRECACHE), 'prepare-graphics', str(store), str(manifest), str(output)]
            subprocess.run(command, check=True)
            cache = output / '_PipelineCache'
            lines = (cache / 'fixture_1.graphics.kpm').read_text().splitlines()
            self.assertEqual(len(lines), 5)
            vs, ps = [next(line for line in lines if f' {stage} ' in line) for stage in ('vs', 'ps')]
            self.assertTrue(vs.endswith('cur=33 gg clip=1 wave=32'))
            self.assertTrue(ps.endswith('cur=19 gg'))
            prefix = f'P v={vs.split()[1]} p={ps.split()[1]}'
            self.assertEqual(set(lines[-2:]), {prefix + ' c0=R8Unorm', prefix + ' c0=R8Uint plain'})
            self.assertEqual((cache / 'fixture_1.code').read_bytes(), store.read_bytes()[:-17])
            missing = root / 'missing'
            store.write_bytes(store.read_bytes()[:25])
            self.assertNotEqual(subprocess.run(command[:-1] + [str(missing)], capture_output=True).returncode, 0)
            self.assertFalse(missing.exists())

    def test_selects_shipped_compute_gaps_and_ranks_complete_reflection(self):
        texture_free, legacy_free, textured, legacy_textured, unknown, existing, graphics, no_recipe = (
            f'{i:016x}' for i in range(1, 9)
        )
        passes = [
            {'cs': {'code': unknown, 'reflected_texture_count': 0}},
            {'cs': {'code': unknown}},
            {'cs': {'code': textured, 'reflected_texture_count': 0}},
            {'cs': {'code': textured, 'reflected_texture_count': 2}},
            {'cs': {'code': legacy_textured, 'resources': [{'kind': 2, 'category': 0, 'texture_dimension': None}]}},
            {'cs': {'code': texture_free, 'reflected_texture_count': 0}},
            {'cs': {'code': legacy_free, 'resources': []}},
            {'cs': {'code': existing, 'reflected_texture_count': 0}},
            {'vs': {'code': graphics, 'reflected_texture_count': 0}},
            {'cs': {'code': no_recipe, 'reflected_texture_count': 0}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / 'graphics-passes.json'
            recipes = root / 'compute-recipes.txt'
            manifest = root / 'existing.kpm'
            selection = root / 'selection.txt'
            inventory.write_text(json.dumps({'bundles': [{'effects': [{'passes': [
                {'shaders': stages} for stages in passes
            ]}]}]}))
            recipes.write_text('kyty compute recipes 1\n' + ''.join(
                f'{digest} 32 1 1 0 0 0 0\n' for digest in
                (unknown, textured, legacy_textured, texture_free, legacy_free, existing, graphics)
            ))
            manifest.write_text(f'# kyty pipeline manifest v1\nM g vs code={texture_free}\nM c cs code={existing}\n')
            command = [sys.executable, str(PRECACHE), '--binary', str(root / 'missing'), 'select',
                       str(inventory), str(recipes), '--existing', str(manifest)]
            all_gaps = subprocess.check_output(command, text=True)
            self.assertEqual(all_gaps.splitlines(), [texture_free, legacy_free, legacy_textured, textured, unknown])
            subprocess.run([*command, '--texture-free', '--limit', '1', '--output', str(selection)], check=True)
            self.assertEqual(selection.read_text(), f'{texture_free}\n')


if __name__ == '__main__':
    unittest.main()
