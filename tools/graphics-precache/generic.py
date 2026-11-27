import argparse
import tempfile
from pathlib import Path

from bind import RESOURCES
from catalog import xxhash
from interfaces import STAGE_CURSOR
from recipe_coverage import remap_pipeline
from session import pipeline_modules, read_manifest, subset_store


def prepare_graphics(store, signatures, output):
    with signatures.open() as source:
        if source.readline().rstrip() != "# kyty pipeline manifest v1":
            raise ValueError("unsupported pipeline manifest")
    modules, pipelines = read_manifest(signatures)
    hash64 = xxhash()
    mapping = {}
    shapes = {}
    codes = set()
    for ident, (stage, fields, _) in modules.items():
        if stage not in STAGE_CURSOR:
            continue
        common = [f"code={fields['code']}"]
        codes.add(fields["code"])
        if "back" in fields:
            common.append(f"back={fields['back']}")
            codes.add(fields["back"])
        if "ud" in fields:
            common.append(f"ud={fields['ud']}")
        common.extend((f"cur={STAGE_CURSOR[stage]}", "gg"))
        static = [f"{k}={v}" if v else k for k, v in fields.items()
                  if k not in (*RESOURCES, "code", "back", "ud", "cur")]
        shape = " ".join([stage, *common, *static])
        raw = shape.encode()
        new_id = f"{hash64(raw, len(raw)) >> 16:012x}"
        if shapes.setdefault(new_id, shape) != shape:
            raise ValueError("generic module identifier collision")
        mapping[ident] = new_id
    if not shapes:
        raise ValueError("no graphics module recipes")
    ready = set()
    unsupported = 0
    for line in pipelines:
        stages = pipeline_modules(line)
        if not stages or any(s not in mapping for s in stages):
            unsupported += 1
            continue
        ready.add(remap_pipeline(line, mapping))
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp) / "prepared"
        cache = root / "_PipelineCache"
        cache.mkdir(parents=True)
        subset_store(store, cache / "offline_1.code", codes)
        with (cache / "offline_1.kpm").open("x") as manifest:
            print("# kyty pipeline manifest v1", file=manifest)
            for ident, shape in sorted(shapes.items()):
                print("M", ident, shape, file=manifest)
            for line in sorted(ready):
                print(line, file=manifest)
        root.rename(output)
    print(f"{len(mapping)} stage requests, {len(shapes)} generic modules, "
          f"{len(pipelines) - unsupported} pipeline requests, {len(ready)} signatures, "
          f"{unsupported} unsupported or unlinked pipelines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("signatures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare_graphics(args.store, args.signatures, args.output)
