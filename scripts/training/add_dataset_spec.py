#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--path", required=True)
parser.add_argument(
    "--type",
    choices=("prepared_folder", "indexed_prepared_tar", "webdataset_text_to_image"),
    default="prepared_folder",
)
parser.add_argument("--index-path")
parser.add_argument("--shards", nargs="+")
parser.add_argument("--weight", required=True, type=float)
parser.add_argument("--samples_per_epoch", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

path = Path(args.config)
config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
    "samples_per_epoch": args.samples_per_epoch,
    "seed": args.seed,
    "shuffle": True,
    "datasets": [],
}
datasets = config.setdefault("datasets", [])
datasets[:] = [dataset for dataset in datasets if dataset.get("name") != args.name]
entry = {"name": args.name, "weight": args.weight}
if args.type == "prepared_folder":
    entry["path"] = str(Path(args.path).resolve())
elif args.type == "indexed_prepared_tar":
    if not args.index_path:
        parser.error("--index-path is required for --type indexed_prepared_tar")
    entry.update(
        type=args.type,
        path=str(Path(args.path).resolve()),
        index_path=str(Path(args.index_path).resolve()),
    )
else:
    shards = args.shards or [args.path]
    entry.update(type=args.type, shards=shards[0] if len(shards) == 1 else shards)
datasets.append(entry)
if any(float(dataset.get("weight", 0)) <= 0 for dataset in datasets):
    raise ValueError("All dataset weights must be positive")
path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
