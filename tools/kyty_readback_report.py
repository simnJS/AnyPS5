import argparse
from collections import Counter, defaultdict
from pathlib import Path


DETAIL = "[kyty-readback-detail] "
REPLAY = "[kyty-readback-replay] "
COLUMNS = {
    "caller": "total_ns",
    "cp_order": "ordered_ns",
    "dispatch_q": "dispatch_queue_ns",
    "dispatch_cb": "dispatch_work_ns",
    "dispatch_call": "dispatch_call_ns",
    "timeline": "timeline_ns",
    "fence": "fence_ns",
    "priority": "priority_ns",
    "publish": "publish_ns",
    "release_q": "release_queue_ns",
    "release_cb": "release_work_ns",
    "release_call": "release_call_ns",
}


def writer(row):
    dirty = int(row["dirty_bytes"])
    if not dirty:
        return "no dirty bytes"
    if int(row["shader_bytes"]) == dirty:
        return "shader"
    if int(row["immediate_bytes"]) == dirty:
        return "immediate"
    if int(row["shader_bytes"]) + int(row["immediate_bytes"]) < dirty:
        return "includes unclassified bytes"
    return "mixed"


def read(log):
    requests = {}
    replay = {}
    for number, line in enumerate(log, 1):
        marker = next((key for key in (DETAIL, REPLAY) if key in line), None)
        if marker is None:
            continue
        try:
            fields = dict(item.split("=", 1) for item in line.split(marker, 1)[1].split())
            ident = int(fields["id"])
            dest = requests if marker == DETAIL else replay
            if ident in dest:
                raise ValueError(f"duplicate {marker.strip()} id {ident}")
            dest[ident] = fields if marker == DETAIL else int(fields["wait_ns"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"line {number}: {error}") from error
    return requests, replay


def main():
    parser = argparse.ArgumentParser(description="Rank readback costs in one process log. Times are cumulative seconds.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--thread", choices=["cp", "other"])
    args = parser.parse_args()
    with args.log.open(errors="replace") as log:
        rows, replay = read(log)
    if not rows:
        parser.exit(1, "No completed readback detail records.\n")
    groups = defaultdict(lambda: defaultdict(Counter))
    for ident, row in rows.items():
        if args.thread and row["thread"] != args.thread:
            continue
        times = {name: int(row[column]) for name, column in COLUMNS.items()}
        times["join"] = int(row.get("join_ns", 0))
        times["n"] = 1
        times["replay_wait"] = replay.get(ident, 0)
        times["caller"] += times["replay_wait"]
        times["async"] = int(row["async"] == "true")
        source = writer(row)
        location = "requested bytes" if row["request_dirty"] == "true" else "other bytes on page" if int(row["dirty_bytes"]) else "clean page"
        if row["path"] in ("joined_cp", "clean_fault"):
            source = location = "not sampled"
        keys = {
            "Thread": row["thread"],
            "Access": f"{row['access']} / {location}",
            "Block": row["block"],
            "Page": row["page"],
            "Writer on requested pages": source,
            "Writer tick on requested pages": f"{source} / {row['writer_tick']} / pending={row['pending']} / unknown={row['unknown_tick']}",
            "Path": row["path"],
        }
        for group, key in keys.items():
            groups[group][key].update(times)
    if not groups:
        parser.exit(1, "No completed records for the selected thread.\n")
    columns = ["n", "async", *COLUMNS, "join", "replay_wait"]
    for label, group in groups.items():
        print(f"\n{label}  (cumulative seconds)")
        print("key\t" + "\t".join(columns))
        for key, counts in sorted(group.items(), key=lambda item: -item[1]["caller"])[:args.top]:
            cells = [str(counts[c]) if c in ("n", "async") else f"{counts[c] / 1e9:.6f}" for c in columns]
            print(f"{key}\t" + "\t".join(cells))
    unmatched = replay.keys() - rows.keys()
    if unmatched:
        print(f"\n{len(unmatched)} replay waits lack a completed detail record.")


if __name__ == "__main__":
    main()
