#!/usr/bin/env python3
"""Write per-source image/mask composition counts for prepared members."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--member", action="append", default=[], metavar="NAME=DIR",
        help="Prepared member to inspect; may be passed multiple times.",
    )
    args = parser.parse_args()

    report: dict[str, object] = {"schema_version": 1, "members": {}}
    for value in args.member:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"--member must be NAME=DIR, got {value!r}")
        member = Path(raw_path).expanduser().resolve()
        metadata = member / "metadata.jsonl"
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        sources: dict[str, dict[str, object]] = {}
        with metadata.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = str(row.get("source_dataset") or "unknown")
                entry = sources.setdefault(source, {
                    "prepared_rows": 0,
                    "conditioning_images": set(),
                    "target_masks": set(),
                    "loss_masks": set(),
                    "splits": collections.Counter(),
                    "groups": collections.Counter(),
                })
                entry["prepared_rows"] = int(entry["prepared_rows"]) + 1
                for field, key in (("conditioning_image", "conditioning_images"),
                                   ("image", "target_masks"), ("loss_mask", "loss_masks")):
                    if row.get(field) not in (None, ""):
                        entry[key].add(str(row[field]))  # type: ignore[index]
                entry["splits"][str(row.get("split") or "unknown")] += 1  # type: ignore[index]
                group = (row.get("metadata") or {}).get("group")
                if group not in (None, ""):
                    entry["groups"][str(group)] += 1  # type: ignore[index]

        rendered = {}
        for source, entry in sorted(sources.items()):
            rendered[source] = {
                "prepared_rows": entry["prepared_rows"],
                "conditioning_images": len(entry["conditioning_images"]),
                "target_masks": len(entry["target_masks"]),
                "loss_masks": len(entry["loss_masks"]),
                "splits": dict(sorted(entry["splits"].items())),
                "groups": dict(sorted(entry["groups"].items())),
            }
        report["members"][name] = {"metadata": str(metadata), "sources": rendered}  # type: ignore[index]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote prepared dataset composition report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
