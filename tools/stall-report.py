#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import struct
import sys


CODE_STORE_MAGIC = 0x3145444F4359544B
STRIDE_MASK = 0x3FFF
BOUNDS_BITS = (1 << 21) | (1 << 22)
VARIATION_FIELDS = ("sb", "si", "ss")
GRAPHICS_STAGES = ("vs", "ps", "mesh", "hs", "ds", "ls", "gs")
PIPELINE_STAGE_KEYS = ("v", "h", "d", "p")
PIPELINE_FLOAT_KEYS = ("fv", "fh", "fd", "fp")
FAMILIES = (
    ("unknown-program", "program never seen (not in the code store)"),
    ("generic-refused", "generic form refused"),
    ("generic-not-prepared", "generic form accepted but not prepared"),
    ("generic-unbound", "generic form prepared but not bound"),
    ("library-missing", "pipeline library missing for this render state"),
    ("other-key", "recipe under another program key"),
    ("other", "other"),
)
LIBRARY_FIELDS = {
    "vi": ("topo", "restart", "vb", "va"),
    "pr": ("neg", "noclip", "pcp", "mhull", "pvlast", "poly"),
    "fs": ("ms", "sshade", "a2c"),
}

BUFFER_AXES = {"f": "buffer-format", "sw": "buffer-format", "oob": "buffer-empty", "empty": "buffer-empty",
               "ir": "buffer-table", "im": "buffer-table", "is": "buffer-table"}
IMAGE_AXES = {"n": "image-class", "d": "image-dimension", "cube": "image-dimension", "tc": "image-table-capacity",
              "t": "image-table", "tm": "image-table", "te": "image-table", "heap": "image-heap",
              "ir": "image-search", "im": "image-search", "is": "image-search"}
KEY_FLAGS = {"rb": "key-runtime-buffers", "rs": "key-runtime-samplers", "async": "key-async-queue"}
AXIS_GROUPS = (
    ("image-addressing", ("image-search", "image-search-classes", "image-search-candidates", "image-heap",
                          "image-table", "image-table-capacity")),
    ("buffer-bounds-bits", ("buffer-bounds-bits",)),
    ("buffer-layout", ("buffer-count", "buffer-stride", "buffer-format", "buffer-empty", "buffer-table")),
    ("image-class", ("image-class",)),
    ("image-shape", ("image-count", "image-dimension", "image-format")),
    ("samplers", ("sampler-count", "sampler-table")),
    ("program-key", ("key-runtime-buffers", "key-runtime-samplers", "key-async-queue", "key-static-interface",
                     "push-data-cursor")),
    ("function-code", ("function-code",)),
)
AXIS_GROUP = {axis: group for group, axes in AXIS_GROUPS for axis in axes}
STATE_AXES = (
    ("color-format", re.compile(r"cc|c\d")),
    ("depth-format", re.compile(r"depth|stencil")),
    ("color-mask", re.compile(r"m\d")),
    ("blend", re.compile(r"b\d")),
    ("samples", re.compile(r"ms|sshade|a2c")),
    ("rasterizer", re.compile(r"neg|noclip|pvlast|poly|pcp|mhull")),
    ("vertex-input", re.compile(r"topo|restart|vb|va")),
)

STALL = re.compile(r"\[kyty-gpl\] stall (\d+\.\d+) s (\S+) 0x([0-9a-fA-F]{16})(?: (.*?))? ?@(\d+)\s*$")
TRANSLATE = re.compile(r"\[kyty-gpl\] translate (\d+\.\d+) s 0x([0-9a-fA-F]{16})(?: (.*?))? ?@(\d+)\s*$")
WAITING = re.compile(r"\[kyty-gpl\] waiting_shader shader=([0-9a-fA-F]{16}) elapsed_s=(\d+\.\d+) monotonic_ns=(\d+)")
GUEST_ENTRY = re.compile(r"\[kyty-startup\] guest_entry monotonic_ns=(\d+)")
GENERIC_REFUSED = re.compile(r"\[kyty-startup\] generic_refused stage=\d+ program=([0-9a-fA-F]{16}) variant=\S+ reason=(\S+)")
UNRESOLVED_MODULE = re.compile(
    r"\[kyty-startup\] unresolved_module stage=\d+ program=([0-9a-fA-F]{16}) variant=\S+"
    r"(?: generic=(\d))? reason=(\S+)(?: detail=(\S+))?")
UNRESOLVED_COMPUTE = re.compile(r"\[kyty-startup\] unresolved_compute shader=([0-9a-fA-F]{16}) variant=\S+ reason=(\S+)")
MANIFEST_WRITTEN = "Pipeline manifest: wrote "
MANIFEST_LOADED = re.compile(r"Pipeline manifest: (\d+) modules, (\d+) pipelines in (\S+) \(loaded")
CODE_LOADED = re.compile(r"Pipeline manifest: the code of \d+ shaders comes from (\S+) \((\d+) shaders\)")
RUNTIME_DESCRIPTORS = re.compile(r"\[kyty\] (?:compute )?shader 0x([0-9a-fA-F]{16}): its (buffer|sampler) descriptors chosen in memory")
ROUTE_FIELD = re.compile(r"\b(why|cause|variant)=(\S+)")
LIBRARY_STATE = re.compile(r"\b(fs|pr)=(queued|building|ready)\b")


@dataclass(slots=True)
class Module:
    ident: str
    stage: str
    code: str
    fields: dict
    generic: bool

    def program_key(self):
        return (self.stage, self.code) + tuple(
            (key, value) for key, value in sorted(self.fields.items())
            if key not in VARIATION_FIELDS and key not in ("gg", "fn") and (key != "cur" or self.stage == "cs"))

    def recipe_key(self, canonical):
        return self.program_key() + (self.generic, self.fields.get("fn", "")) + tuple(
            canonical_elements(key, self.fields.get(key, "")) if canonical else self.fields.get(key, "")
            for key in VARIATION_FIELDS)

    def keeps_bounds_in_key(self):
        text = self.fields.get("sb", "")
        return isinstance(text, str) and canonical_elements("sb", text) != text


@dataclass(slots=True)
class Pipeline:
    stages: dict
    state: tuple


@dataclass(slots=True)
class Manifest:
    modules: list = field(default_factory=list)
    pipelines: list = field(default_factory=list)
    by_id: dict = field(default_factory=dict)


@dataclass(slots=True)
class Wait:
    kind: str
    program: str
    seconds: float
    end_ns: int
    route: str = ""
    why: str = ""
    stage: str = "?"
    variant: str = ""
    cause: str = ""
    evidence: str = ""
    engine_cause: bool = False
    announced: bool = False
    known: bool = True
    line: int = 0

    @property
    def family(self):
        return self.cause.partition(":")[0]


def parse_fields(tokens):
    fields = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        fields[key] = value if separator else True
    return fields


def split_elements(text):
    return text.split(",") if text else []


def element_fields(element):
    if element in ("", "-"):
        return {}
    fields = {}
    for part in element.split("/"):
        key, separator, value = part.partition(":")
        fields[key] = value if separator else True
    return fields


def canonical_elements(key, text):
    if key != "sb" or not text:
        return text
    elements = []
    for element in text.split(","):
        fields = element_fields(element)
        stride = fields.get("st")
        if isinstance(stride, str) and stride.isdigit() and "ir" not in fields and int(stride) & STRIDE_MASK == STRIDE_MASK:
            fields["st"] = str(int(stride) & ~BOUNDS_BITS)
            element = "/".join(name if value is True else f"{name}:{value}" for name, value in fields.items())
        elements.append(element)
    return ",".join(elements)


ELEMENT_AXES = {}


def element_axes(kind, element, canonical=False):
    cached = ELEMENT_AXES.get((kind, element, canonical))
    if cached is not None:
        return cached
    axes = defaultdict(list)
    names = BUFFER_AXES if kind == "buffer" else IMAGE_AXES
    fields = element_fields(element)
    for name, value in sorted(fields.items()):
        if kind == "buffer" and name == "st":
            stride = int(value) if isinstance(value, str) and value.isdigit() else 0
            read_at_run_time = canonical and "ir" not in fields and stride & STRIDE_MASK == STRIDE_MASK
            if stride & BOUNDS_BITS and not read_at_run_time:
                axes["buffer-bounds-bits"].append(stride & BOUNDS_BITS)
            axes["buffer-stride"].append(stride & ~BOUNDS_BITS)
        else:
            axes[names.get(name, f"{kind}-format")].append((name, value))
    cached = {axis: tuple(values) for axis, values in axes.items()}
    ELEMENT_AXES[(kind, element, canonical)] = cached
    return cached


def recipe_axes(module, canonical=False):
    axes = {}
    static = []
    for name, value in sorted(module.fields.items()):
        if name in KEY_FLAGS:
            axes[KEY_FLAGS[name]] = True
        elif name == "fn":
            axes["function-code"] = value
        elif name == "cur":
            if module.stage == "cs":
                axes["push-data-cursor"] = value
        elif name not in VARIATION_FIELDS and name != "gg":
            static.append((name, value))
    axes["key-static-interface"] = tuple(static)
    buffers = split_elements(module.fields.get("sb", ""))
    axes["buffer-count"] = len(buffers)
    per_axis = defaultdict(list)
    for index, element in enumerate(buffers):
        for axis, value in element_axes("buffer", element, canonical).items():
            per_axis[axis].append((index, value))
    images = split_elements(module.fields.get("si", ""))
    counts = Counter(images)
    roots = set()
    plain = set()
    for element in counts:
        root = element_fields(element).get("ir")
        if root is None:
            plain.add(element)
        elif isinstance(root, str) and root.isdigit() and int(root) < len(images) and images[int(root)] == element:
            roots.add(int(root))
    if plain:
        roots.update(index for index, element in enumerate(images) if element in plain)
    for index in roots:
        counts[images[index]] -= 1
    candidates = tuple(sorted((element, count) for element, count in counts.items() if count > 0))
    axes["image-count"] = len(roots)
    if candidates:
        axes["image-search-candidates"] = sum(count for _, count in candidates)
        axes["image-search-classes"] = candidates
    for position, index in enumerate(sorted(roots)):
        for axis, value in element_axes("image", images[index]).items():
            per_axis[axis].append((position, value))
    for axis, values in per_axis.items():
        axes[axis] = tuple(values)
    samplers = split_elements(module.fields.get("ss", ""))
    axes["sampler-count"] = len(samplers)
    if any(element not in ("", "-") for element in samplers):
        axes["sampler-table"] = tuple(samplers)
    return axes


def state_axes(state):
    axes = defaultdict(list)
    for name, value in state:
        axis = next((label for label, pattern in STATE_AXES if pattern.fullmatch(name)), "other-state")
        axes[axis].append((name, value))
    return {axis: tuple(values) for axis, values in axes.items()}


def axis_table(groups, grouping=None):
    table = defaultdict(lambda: {"groups": 0, "values": 0, "merged": 0})
    for recipes in groups:
        distinct = {tuple(sorted(recipe.items(), key=lambda item: item[0])) for recipe in map(freeze_axes, recipes)}
        if len(distinct) < 2:
            continue
        names = defaultdict(set)
        for recipe in distinct:
            for name, _ in recipe:
                names[grouping.get(name, name) if grouping else name].add(name)
        for label, members in names.items():
            values = {tuple(item for item in recipe if item[0] in members) for recipe in distinct}
            if len(values) < 2:
                continue
            without = {tuple(item for item in recipe if item[0] not in members) for recipe in distinct}
            row = table[label]
            row["groups"] += 1
            row["values"] += len(values) - 1
            row["merged"] += len(distinct) - len(without)
    return dict(sorted(table.items(), key=lambda item: (-item[1]["values"], item[0])))


def freeze_axes(axes):
    return {name: value if isinstance(value, (tuple, str, int, bool)) else repr(value) for name, value in axes.items()}


def read_manifest(path, module_limit=None, pipeline_limit=None):
    manifest = Manifest()
    if path is None or not Path(path).is_file():
        return manifest
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("M "):
                if module_limit is not None and len(manifest.modules) >= module_limit:
                    continue
                tokens = line.split()
                if len(tokens) < 4:
                    continue
                fields = parse_fields(tokens[3:])
                code = fields.pop("code", "")
                if not isinstance(code, str):
                    continue
                module = Module(tokens[1], tokens[2], code.lower(), fields, "gg" in fields)
                manifest.modules.append(module)
                manifest.by_id.setdefault(module.ident, module)
            elif line.startswith("P "):
                if pipeline_limit is not None and len(manifest.pipelines) >= pipeline_limit:
                    continue
                fields = parse_fields(line.split()[1:])
                stages = {key: fields.pop(key) for key in PIPELINE_STAGE_KEYS if key in fields}
                for key in PIPELINE_FLOAT_KEYS:
                    fields.pop(key, None)
                fields.pop("plain", None)
                manifest.pipelines.append(Pipeline(stages, tuple(sorted(fields.items(), key=lambda item: item[0]))))
    return manifest


def read_code_store(path, limit=None):
    programs = []
    if path is None or not Path(path).is_file():
        return None
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if stream.read(8) != struct.pack("<Q", CODE_STORE_MAGIC):
            raise ValueError(f"{path}: unsupported code store")
        while header := stream.read(16):
            if len(header) != 16:
                break
            digest, words, compressed = struct.unpack("<QII", header)
            if not words or not compressed or compressed > size - stream.tell():
                break
            stream.seek(compressed, os.SEEK_CUR)
            programs.append(f"{digest:016x}")
    if limit is not None and len(programs) >= limit:
        programs = programs[:limit]
    return set(programs)


def diff_elements(key, wanted, prepared):
    wanted_elements = split_elements(wanted)
    prepared_elements = split_elements(prepared)
    names = {"sb": "buffer", "si": "image", "ss": "sampler"}
    if len(wanted_elements) != len(prepared_elements):
        return 1000 + abs(len(wanted_elements) - len(prepared_elements)), {
            f"{names[key]}-count({len(prepared_elements)}->{len(wanted_elements)})"}
    reasons = set()
    distance = 0
    for wanted_element, prepared_element in zip(wanted_elements, prepared_elements):
        if wanted_element == prepared_element:
            continue
        distance += 1
        left = element_fields(wanted_element)
        right = element_fields(prepared_element)
        for name in sorted(set(left) | set(right)):
            if left.get(name) == right.get(name):
                continue
            if key == "sb" and name == "st":
                try:
                    changed = int(left.get(name, 0)) ^ int(right.get(name, 0))
                except (TypeError, ValueError):
                    changed = -1
                reasons.add("buffer-bounds-bits" if changed >= 0 and changed & ~BOUNDS_BITS == 0 else "buffer-stride")
            elif key == "si" and name == "tc":
                reasons.add(f"image-table-capacity({right.get(name, 0)}->{left.get(name, 0)})")
            else:
                reasons.add(f"{names[key]}-{name}")
    return distance, reasons


def closest_recipe(wanted, prepared, canonical):
    best = None
    for candidate in prepared:
        distance = 0
        reasons = set()
        for key in VARIATION_FIELDS:
            left = wanted.fields.get(key, "")
            right = candidate.fields.get(key, "")
            if canonical:
                left, right = canonical_elements(key, left), canonical_elements(key, right)
            step, why = diff_elements(key, left, right)
            distance += step
            reasons |= why
        if wanted.fields.get("fn", "") != candidate.fields.get("fn", ""):
            distance += 1
            reasons.add("function-code")
        if best is None or distance < best[0]:
            best = (distance, reasons, candidate)
    return best


def key_difference(wanted, prepared):
    best = None
    for candidate in prepared:
        left = dict(wanted.program_key()[2:])
        right = dict(candidate.program_key()[2:])
        names = sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
        if best is None or len(names) < len(best):
            best = names
    return best or []


class Session:
    def __init__(self, log, session_manifest=None, shared_manifest=None, code_store=None):
        self.log = Path(log)
        self.guest_entry = None
        self.refused = {}
        self.unresolved = defaultdict(Counter)
        self.runtime_descriptors = defaultdict(set)
        self.waits = []
        self.unfinished = []
        shared_counts = None
        code_count = None
        announcements = []
        self.last_write = 0
        with self.log.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if "[kyty-gpl] " in line:
                    if match := STALL.search(line):
                        route = (match[4] or "").strip()
                        extra = dict(ROUTE_FIELD.findall(route))
                        self.waits.append(Wait(match[2], match[3].lower(), float(match[1]), int(match[5]),
                                               ROUTE_FIELD.sub("", route).strip(), extra.get("why", ""),
                                               variant=extra.get("variant", ""), cause=extra.get("cause", ""),
                                               engine_cause="cause" in extra, line=number))
                    elif match := TRANSLATE.search(line):
                        extra = dict(ROUTE_FIELD.findall(match[3] or ""))
                        self.waits.append(Wait("translate", match[2].lower(), float(match[1]), int(match[4]),
                                               variant=extra.get("variant", ""), cause=extra.get("cause", ""),
                                               engine_cause="cause" in extra, line=number))
                    elif match := WAITING.search(line):
                        announcements.append((match[1].lower(), float(match[2]), int(match[3]), number))
                elif line.startswith(MANIFEST_WRITTEN):
                    self.last_write = number
                elif "[kyty-startup] " in line:
                    if match := GENERIC_REFUSED.search(line):
                        self.refused.setdefault(match[1].lower(), match[2])
                    elif match := UNRESOLVED_MODULE.search(line):
                        self.unresolved[match[1].lower()][match[4] or match[3]] += 1
                    elif match := UNRESOLVED_COMPUTE.search(line):
                        self.unresolved[match[1].lower()][match[2]] += 1
                    elif match := GUEST_ENTRY.search(line):
                        self.guest_entry = int(match[1])
                elif line.startswith("Pipeline manifest: "):
                    if (match := MANIFEST_LOADED.search(line)) and match[3].endswith(".shared.kpm"):
                        shared_counts = (int(match[1]), int(match[2]))
                    elif match := CODE_LOADED.search(line):
                        code_count = int(match[2])
                elif match := RUNTIME_DESCRIPTORS.search(line):
                    self.runtime_descriptors[match[1].lower()].add("rb" if match[2] == "buffer" else "rs")
        for program, elapsed, at, number in announcements:
            started = at - int(elapsed * 1e9)
            owner = next((wait for wait in self.waits
                          if wait.program == program and wait.kind != "translate" and wait.end_ns >= at and
                          wait.end_ns - int(wait.seconds * 1e9) <= started + 500_000_000), None)
            if owner is not None:
                owner.announced = True
            else:
                self.waits.append(Wait("unfinished", program, elapsed, at, announced=True, line=number))
        self.waits.sort(key=lambda wait: wait.end_ns)

        self.shared_loaded = shared_counts
        self.prepared = read_manifest(shared_manifest, *(shared_counts or (None, None)))
        self.recorded = read_manifest(session_manifest)
        self.known = read_code_store(code_store, code_count)
        self.prepared_programs = defaultdict(list)
        for module in self.prepared.modules:
            self.prepared_programs[module.code].append(module)
        self.canonical_strides = not any(module.keeps_bounds_in_key() for module in self.recorded.modules)
        prepared_recipes = {module.recipe_key(self.canonical_strides) for module in self.prepared.modules}
        self.new_programs = defaultdict(list)
        self.new_ids = set()
        for module in self.recorded.modules:
            if not module.generic and module.recipe_key(self.canonical_strides) not in prepared_recipes:
                self.new_programs[module.code].append(module)
                self.new_ids.add(module.ident)
        self.stages = {}
        for module in self.recorded.modules + self.prepared.modules:
            self.stages.setdefault(module.code, module.stage)
        self.prepared_states = set()
        for pipeline in self.prepared.pipelines:
            programs = self.pipeline_programs(self.prepared, pipeline)
            if programs is not None:
                self.prepared_states.add((programs, pipeline.state))
        self.prepared_pipelines_by_stages = defaultdict(list)
        for pipeline in self.prepared.pipelines:
            programs = self.pipeline_programs(self.prepared, pipeline)
            if programs is not None:
                self.prepared_pipelines_by_stages[programs].append(pipeline)
        self.new_pipelines = defaultdict(list)
        seen = set()
        for pipeline in self.recorded.pipelines:
            programs = self.pipeline_programs(self.recorded, pipeline)
            identity = (tuple(sorted(pipeline.stages.items())), pipeline.state)
            if programs is None or identity in seen:
                continue
            seen.add(identity)
            fresh_modules = any(ident in self.new_ids for ident in pipeline.stages.values())
            if fresh_modules or (programs, pipeline.state) not in self.prepared_states:
                for key in pipeline.stages:
                    module = self.recorded.by_id.get(pipeline.stages[key])
                    if module is not None:
                        self.new_pipelines[module.code].append(pipeline)
        for wait in self.waits:
            self.classify(wait)

    @staticmethod
    def pipeline_programs(manifest, pipeline):
        programs = []
        for key in PIPELINE_STAGE_KEYS:
            ident = pipeline.stages.get(key)
            if ident is None:
                continue
            module = manifest.by_id.get(ident)
            if module is None:
                return None
            programs.append((key,) + module.program_key())
        return tuple(programs)

    def is_known(self, program):
        if self.known is not None:
            return program in self.known
        return program in self.prepared_programs

    def variant_cause(self, wanted, compute):
        prepared = [module for module in self.prepared_programs.get(wanted.code, []) if not module.generic]
        same_key = [module for module in prepared if module.program_key() == wanted.program_key()]
        if not same_key:
            names = key_difference(wanted, prepared)
            return "other-key:" + ("+".join(names) if names else "static-interface"), (
                f"variant {wanted.ident} has no prepared recipe under its program key; "
                f"{len(prepared)} recipes exist under another key")
        distance, reasons, closest = closest_recipe(wanted, same_key, self.canonical_strides)
        unresolved = self.unresolved.get(wanted.code)
        if unresolved and any(reason.endswith("-count") or "-count(" in reason for reason in reasons):
            detail = "+".join(sorted(unresolved))
            return f"generic-refused:{detail}", (
                f"variant {wanted.ident}: the {sum(unresolved.values())} prepared recipes of this program were not "
                f"rebuilt at startup; closest recipe {closest.ident} differs by {'+'.join(sorted(reasons))}")
        if not reasons:
            reasons = {"identical-recipe"}
        return "generic-unbound:" + "+".join(sorted(reasons)), (
            f"variant {wanted.ident} against its closest prepared recipe {closest.ident}")

    def library_cause(self, wait, pipelines):
        missing = [part for part, state in LIBRARY_STATE.findall(wait.route) if state != "ready"]
        fields = set()
        for pipeline in pipelines:
            programs = self.pipeline_programs(self.recorded, pipeline)
            candidates = self.prepared_pipelines_by_stages.get(programs, [])
            if not candidates:
                fields.add("stage-pair")
                continue
            state = dict(pipeline.state)
            best = None
            for candidate in candidates:
                other = dict(candidate.state)
                names = {name for name in set(state) | set(other) if state.get(name) != other.get(name)}
                if best is None or len(names) < len(best):
                    best = names
            fields |= best or set()
        parts = set(missing)
        for name in fields:
            for part, names in LIBRARY_FIELDS.items():
                if name in names:
                    parts.add(part)
            if name not in sum(LIBRARY_FIELDS.values(), ()) and name != "stage-pair":
                parts.add("fo")
        order = ("vi", "pr", "fs", "fo")
        label = "+".join(part for part in order if part in parts) or "unspecified"
        evidence = "render state differs from the prepared pipelines by " + (", ".join(sorted(fields)) or "nothing recorded")
        if missing:
            evidence += "; libraries built during the wait: " + "+".join(missing)
        return f"library-missing:{label}", evidence

    def classify(self, wait):
        program = wait.program
        wait.stage = "cs" if wait.kind.startswith("compute") else "ps" if wait.kind.startswith("graphics") else \
            self.stages.get(program, "?")
        wait.known = self.is_known(program)
        fresh = self.new_programs.get(program, [])
        if not wait.variant and len(fresh) == 1:
            wait.variant = fresh[0].ident
        elif not wait.variant and fresh:
            wait.variant = f"{len(fresh)} new variants"
        if wait.engine_cause:
            wait.evidence = "computed by the engine"
            wait.known = wait.family != "unknown-program" and (self.known is None or wait.known)
            return
        if not wait.known:
            wait.cause = "unknown-program"
            wait.evidence = "not in the code store loaded at startup"
            return
        prepared = self.prepared_programs.get(program, [])
        compute = wait.stage == "cs"
        if not compute and program in self.refused:
            wait.cause = f"generic-refused:{self.refused[program]}"
            wait.evidence = "generic_refused at startup"
            return
        if not prepared:
            modes = self.runtime_descriptors.get(program)
            wait.cause = "generic-not-prepared:no-recipe"
            wait.evidence = "known program without any recipe in the manifest loaded at startup" + (
                f" (run-time descriptors: {'+'.join(sorted(modes))})" if modes else "")
            return
        causes = Counter()
        notes = {}
        stage_variants = list(fresh)
        paired = []
        if not compute:
            for pipeline in self.new_pipelines.get(program, []):
                for ident in pipeline.stages.values():
                    module = self.recorded.by_id.get(ident)
                    if module is not None and module.code != program and ident in self.new_ids and module not in paired:
                        paired.append(module)
        for module in paired:
            if module.code in self.refused:
                causes[f"generic-refused:{self.refused[module.code]}"] += 1
                notes.setdefault(f"generic-refused:{self.refused[module.code]}",
                                 f"paired {module.stage} program {module.code} refused at startup")
            else:
                stage_variants.append(module)
        for module in stage_variants:
            cause, note = self.variant_cause(module, compute)
            if not compute and cause.startswith("generic-unbound"):
                pipelines = [pipeline for pipeline in self.new_pipelines.get(module.code, [])
                             if module.ident in pipeline.stages.values()]
                served = [pipeline for pipeline in pipelines
                          if (self.pipeline_programs(self.recorded, pipeline), pipeline.state) in self.prepared_states]
                if pipelines and not served:
                    library, why = self.library_cause(wait, pipelines)
                    cause, note = library, note + "; no generic pipeline prepared for its render state: " + why
            causes[cause] += 1
            notes.setdefault(cause, note + (f" ({module.stage} {module.code})" if module.code != program else ""))
        if causes:
            wait.cause = max(causes, key=lambda cause: (causes[cause], cause))
            wait.evidence = notes[wait.cause]
            if len(causes) > 1:
                wait.evidence += "; other variants of this program: " + ", ".join(
                    f"{cause} x{count}" for cause, count in sorted(causes.items()) if cause != wait.cause)
            return
        pipelines = self.new_pipelines.get(program, []) if not compute else []
        if pipelines:
            wait.cause, wait.evidence = self.library_cause(wait, pipelines)
            return
        if wait.line > self.last_write and self.recorded.modules:
            wait.cause = "other:not-recorded"
            wait.evidence = "the session manifest was last written before this wait, so its variant is not recorded"
        elif wait.kind == "graphics-wait":
            wait.cause = "other:prepared-pipeline-still-building"
            wait.evidence = "modules and render state were in the manifest; the draw waited for a queued creation"
        else:
            wait.cause = "other:prepared-module-without-driver-pipeline"
            wait.evidence = "every recipe requested was in the manifest loaded at startup"

    def variant_map(self, top=20):
        loaded = defaultdict(list)
        fresh = defaultdict(list)
        seen = set()
        for module in self.prepared.modules:
            recipe = module.recipe_key(self.canonical_strides)
            if not module.generic and recipe not in seen:
                seen.add(recipe)
                loaded[(module.stage, module.code)].append(module)
        for modules in self.new_programs.values():
            for module in modules:
                fresh[(module.stage, module.code)].append(module)
        axes = {}

        def axes_of(module):
            found = axes.get(id(module))
            if found is None:
                found = axes[id(module)] = recipe_axes(module, self.canonical_strides)
            return found

        programs = sorted(set(loaded) | set(fresh))
        groups = {program: [axes_of(module) for module in loaded.get(program, []) + fresh.get(program, [])]
                  for program in programs}
        made_new = defaultdict(lambda: {"variants": 0, "alone": 0, "programs": set()})
        for program, modules in fresh.items():
            known = [axes_of(module) for module in loaded.get(program, [])]
            for module in modules:
                wanted = axes_of(module)
                if not known:
                    names = {"no-loaded-recipe"}
                else:
                    names = min(({AXIS_GROUP.get(name, name) for name in set(wanted) | set(other)
                                  if wanted.get(name) != other.get(name)} for other in known), key=len)
                for name in names or {"order-of-elements"}:
                    row = made_new[name]
                    row["variants"] += 1
                    row["alone"] += len(names) <= 1
                    row["programs"].add(program)
        table = axis_table(groups.values())
        grouped = axis_table(groups.values(), AXIS_GROUP)
        ranked = sorted(programs, key=lambda item: -(len(loaded.get(item, [])) + len(fresh.get(item, []))))
        listing = []
        for program in ranked[:top] if top else ranked:
            own = axis_table([groups[program]], AXIS_GROUP)
            listing.append({"stage": program[0], "program": program[1], "loaded": len(loaded.get(program, [])),
                            "new": len(fresh.get(program, [])),
                            "fields": {name: row["values"] + 1 for name, row in own.items()}})
        states = defaultdict(set)
        fresh_states = defaultdict(set)
        for manifest, target in ((self.prepared, states), (self.recorded, fresh_states)):
            for pipeline in manifest.pipelines:
                stages = self.pipeline_programs(manifest, pipeline)
                if stages is not None:
                    target[tuple((stage[1], stage[2]) for stage in stages)].add(pipeline.state)
        new_states = sum(len(found - states.get(pair, set())) for pair, found in fresh_states.items())
        for pair, found in fresh_states.items():
            states[pair] |= found
        state_table = axis_table([[state_axes(state) for state in found] for found in states.values()])
        return {
            "programs": len(programs),
            "stages": dict(Counter(program[0] for program in programs)),
            "loaded": sum(len(modules) for modules in loaded.values()),
            "new": sum(len(modules) for modules in fresh.values()),
            "programs_with_new": len(fresh),
            "field_groups": grouped,
            "fields": table,
            "new_fields": {name: {"variants": row["variants"], "alone": row["alone"], "programs": len(row["programs"])}
                           for name, row in sorted(made_new.items(), key=lambda item: -item[1]["variants"])},
            "top_programs": listing,
            "stage_sets": len(states),
            "render_states": sum(len(found) for found in states.values()),
            "new_render_states": new_states,
            "state_fields": state_table,
        }

    def summary(self):
        families = {name: {"waits": 0, "seconds": 0.0, "programs": set(), "causes": defaultdict(lambda: [0, 0.0, set()])}
                    for name, _ in FAMILIES}
        for wait in self.waits:
            family = families.setdefault(
                wait.family, {"waits": 0, "seconds": 0.0, "programs": set(),
                              "causes": defaultdict(lambda: [0, 0.0, set()])})
            family["waits"] += 1
            family["seconds"] += wait.seconds
            family["programs"].add(wait.program)
            cause = family["causes"][wait.cause]
            cause[0] += 1
            cause[1] += wait.seconds
            cause[2].add(wait.program)
        known = [wait for wait in self.waits if wait.family != "unknown-program"]
        unknown = [wait for wait in self.waits if wait.family == "unknown-program"]
        return {
            "log": str(self.log),
            "waits": len(self.waits),
            "seconds": round(sum(wait.seconds for wait in self.waits), 3),
            "known_waits": len(known),
            "known_seconds": round(sum(wait.seconds for wait in known), 3),
            "unknown_waits": len(unknown),
            "unknown_seconds": round(sum(wait.seconds for wait in unknown), 3),
            "engine_causes": sum(wait.engine_cause for wait in self.waits),
            "known_programs": None if self.known is None else len(self.known),
            "prepared_modules": len(self.prepared.modules),
            "prepared_pipelines": len(self.prepared.pipelines),
            "new_variants": sum(len(modules) for modules in self.new_programs.values()),
            "families": {
                name: {
                    "waits": family["waits"],
                    "seconds": round(family["seconds"], 3),
                    "programs": len(family["programs"]),
                    "causes": {cause: {"waits": values[0], "seconds": round(values[1], 3), "programs": len(values[2])}
                               for cause, values in sorted(family["causes"].items(), key=lambda item: -item[1][1])},
                }
                for name, family in families.items()
            },
        }


def find_inputs(session):
    session = Path(session)
    if session.is_file():
        return session, None, None, None
    cache = session / "_PipelineCache"
    shared = sorted(cache.glob("*.shared.kpm"))
    own = sorted(path for path in cache.glob("*.kpm") if not path.name.endswith(".shared.kpm"))
    code = sorted(cache.glob("*.code"))
    return session / "stderr.log", own[0] if own else None, shared[0] if shared else None, code[0] if code else None


def print_report(session, detail, limit):
    summary = session.summary()
    labels = dict(FAMILIES)
    print(f"log: {summary['log']}")
    known = "unavailable (programs listed by the manifest count as known)" if summary["known_programs"] is None \
        else f"{summary['known_programs']} programs"
    print(f"code store: {known}; manifest loaded at startup: {summary['prepared_modules']} modules, "
          f"{summary['prepared_pipelines']} pipelines; variants new in this session: {summary['new_variants']}")
    source = "computed by the engine" if summary["engine_causes"] == summary["waits"] and summary["waits"] else \
        f"computed by the engine for {summary['engine_causes']} waits, inferred from the manifests for the others"
    print(f"waits: {summary['waits']}, {summary['seconds']:.1f} s; causes {source}")
    print()
    print(f"{'seconds':>9} {'waits':>6} {'programs':>8}  cause")
    for name, family in sorted(summary["families"].items(), key=lambda item: -item[1]["seconds"]):
        print(f"{family['seconds']:9.1f} {family['waits']:6d} {family['programs']:8d}  {labels.get(name, name)}")
        for cause, values in family["causes"].items():
            if cause != name:
                print(f"{values['seconds']:9.1f} {values['waits']:6d} {values['programs']:8d}      {cause}")
    print()
    print(f"waits on known programs: {summary['known_waits']}, {summary['known_seconds']:.1f} s")
    print(f"waits on unknown programs: {summary['unknown_waits']}, {summary['unknown_seconds']:.1f} s")
    if not detail:
        return
    print()
    waits = sorted(session.waits, key=lambda wait: -wait.seconds)[:limit] if limit else session.waits
    for wait in waits:
        at = "" if session.guest_entry is None else f"+{(wait.end_ns - session.guest_entry) / 1e9:8.1f} s "
        extra = " ".join(part for part in (wait.route, f"why={wait.why}" if wait.why else "",
                                           "announced-after-3s" if wait.announced else "") if part)
        print(f"{at}{wait.seconds:8.3f} s {wait.kind:14s} {wait.stage:4s} {wait.program} "
              f"variant={wait.variant or '-'} cause={wait.cause}{' [' + extra + ']' if extra else ''}")
        print(f"{'':12s}{wait.evidence}")


def print_variants(session, top):
    report = session.variant_map(top)
    stages = ", ".join(f"{count} {stage}" for stage, count in sorted(report["stages"].items()))
    print(f"variants: {report['programs']} programs with recipes ({stages}); {report['loaded']} distinct recipes "
          f"loaded at startup, {report['new']} new in this session in {report['programs_with_new']} programs")
    print()
    members = dict(AXIS_GROUPS)
    print("fields that make the variants, over the loaded and the new recipes")
    print(f"{'programs':>9} {'values':>7} {'merged':>7}  field")
    for name, row in report["field_groups"].items():
        print(f"{row['groups']:9d} {row['values']:7d} {row['merged']:7d}  {name}")
        for member in members.get(name, ()):
            detail = report["fields"].get(member)
            if detail is not None and len(members[name]) > 1:
                print(f"{detail['groups']:9d} {detail['values']:7d} {detail['merged']:7d}      {member}")
    print("  values: variants the field makes on its own (its distinct values per program, minus one)")
    print("  merged: recipes that would fold into another one if the field left the key")
    print()
    print("fields that made the new variants of this session, against the closest loaded recipe")
    print(f"{'variants':>9} {'alone':>7} {'programs':>9}  field")
    for name, row in report["new_fields"].items():
        print(f"{row['variants']:9d} {row['alone']:7d} {row['programs']:9d}  {name}")
    print()
    print("programs with the most variants")
    print(f"{'loaded':>7} {'new':>5}  stage program           fields (distinct values)")
    for row in report["top_programs"][:top or 20]:
        fields = ", ".join(f"{name} {count}" for name, count in row["fields"].items())
        print(f"{row['loaded']:7d} {row['new']:5d}  {row['stage']:5s} {row['program']}  {fields}")
    print()
    print(f"render states: {report['render_states']} over {report['stage_sets']} sets of stage programs, "
          f"{report['new_render_states']} new in this session")
    print(f"{'sets':>9} {'values':>7} {'merged':>7}  field")
    for name, row in report["state_fields"].items():
        print(f"{row['groups']:9d} {row['values']:7d} {row['merged']:7d}  {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify every shader and pipeline wait of a session by the reason the generic form did not serve it.")
    parser.add_argument("sessions", nargs="+", type=Path,
                        help="session directory (stderr.log and _PipelineCache) or a log file")
    parser.add_argument("--session-manifest", type=Path)
    parser.add_argument("--shared-manifest", type=Path)
    parser.add_argument("--code-store", type=Path)
    parser.add_argument("--detail", action="store_true", help="list every wait with its evidence")
    parser.add_argument("--top", type=int, default=0, help="with --detail, only the longest N waits")
    parser.add_argument("--variants", action="store_true",
                        help="map the variants: which fields of the key make them, per program and overall")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    results = []
    for index, path in enumerate(arguments.sessions):
        log, own, shared, code = find_inputs(path)
        session = Session(log, arguments.session_manifest or own, arguments.shared_manifest or shared,
                          arguments.code_store or code)
        if arguments.json:
            report = session.summary()
            if arguments.variants:
                report["variants"] = session.variant_map(arguments.top)
            if arguments.detail:
                report["list"] = [
                    {"kind": wait.kind, "stage": wait.stage, "program": wait.program, "seconds": wait.seconds,
                     "end_ns": wait.end_ns, "variant": wait.variant, "cause": wait.cause, "route": wait.route,
                     "why": wait.why, "evidence": wait.evidence, "engine_cause": wait.engine_cause}
                    for wait in session.waits]
            results.append(report)
        else:
            if index:
                print()
            print_report(session, arguments.detail, arguments.top)
            if arguments.variants:
                print()
                print_variants(session, arguments.top or 20)
    if arguments.json:
        json.dump(results, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
