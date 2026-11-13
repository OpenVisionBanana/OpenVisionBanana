#!/usr/bin/env python3
"""Download official raw sources into raw/<Group>/<dataset>/ and explain gated acquisition.

Direct files are fetched with resume. Scripted sources run their acquisition
scripts (download, validate and pack); a gated source's script checks the files
the user placed in its folder and prints the official instructions otherwise.
A training pool expands to the sources it is built from.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.release_data import (
    CATALOG,
    DOWNLOAD_MARKER,
    acquisition_items,
    available_acquisition_items,
    download_with_resume,
    parse_selection,
    raw_directory,
    staged_source_status,
    step_command,
)

# Exit codes that mean a transient network failure: wget 4, curl 6/7/28/35/52/56.
# The acquisition scripts resume or skip finished files, so such a step is retried.
NETWORK_EXIT_CODES = {4, 6, 7, 28, 35, 52, 56}
ATTEMPTS = int(os.environ.get("OVB_DOWNLOAD_ATTEMPTS", "3"))
RETRY_DELAY = int(os.environ.get("OVB_DOWNLOAD_RETRY_DELAY", "60"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    parser.add_argument("--work-dir", type=Path,
                        help="Existing writable directory for temporary files; defaults to the system temporary directory.")
    parser.add_argument("--datasets", default="all", help="Comma-separated names or all.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent direct-file downloads.")
    parser.add_argument("--script-workers", type=int, default=2, help="Concurrent acquisition scripts.")
    parser.add_argument("--list", action="store_true", help="List the release catalog and exit.")
    parser.add_argument("--describe", metavar="DATASET", help="Print official acquisition instructions and exit.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check whether selected official sources have been staged; no downloads are performed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="DATASET.KEY=VALUE",
                        help="Official acquisition setting; repeat as needed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from common.node_scratch import configure_work_dir
    configure_work_dir(getattr(args, "work_dir", None))
    if args.workers < 1 or args.script_workers < 1:
        raise SystemExit("--workers and --script-workers must be positive")
    if args.list:
        for item in CATALOG:
            where = f"raw/{item.raw_group}/{item.name}" if item.raw_group else "sources: " + ",".join(item.sources)
            print(f"{item.name:18} {item.access:10} {where}  {','.join(item.tasks)}")
        return 0
    if args.describe:
        try:
            selected = parse_selection(args.describe)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if len(selected) != 1:
            raise SystemExit("--describe accepts exactly one dataset")
        item = selected[0]
        print(
            f"{item.name}: {item.source}\n"
            f"Access: {item.access}\n"
            f"Tasks: {', '.join(item.tasks)}\n"
            f"Destination: {raw_directory(args.destination, item)}\n"
            f"{item.note}"
        )
        for url, filename in item.files:
            print(f"  {filename}: {url}")
        for step in item.steps:
            print(f"  script: {step.script}")
        if item.expected:
            print("Expected staged paths: " + ", ".join(item.expected))
        return 0

    try:
        selected = acquisition_items(parse_selection(args.datasets))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.status:
        missing = False
        for item in selected:
            ready, details = staged_source_status(args.destination, item)
            state = "READY" if ready else "MISSING"
            suffix = "" if ready else f" ({', '.join(details)})"
            print(f"{state:7} {item.name:18} {raw_directory(args.destination, item)}{suffix}")
            missing |= not ready
        return 1 if missing else 0
    destination = args.destination.expanduser().resolve()
    settings = {}
    for value in args.set:
        left, setting = value.split("=", 1)
        dataset, key = left.split(".", 1)
        if dataset not in {item.name for item in selected}:
            raise SystemExit(f"Download override refers to an unselected source: {dataset}")
        allowed = {"hypersim": {"scenes", "frames"}, "taskonomy": {"num-buildings", "seed", "frames"}, "marigold_normals": {"frames"}, "marigold_depth": {"frames", "datasets"}, "pico_banana": {"limit"}, "pd12m": {"count", "scan-limit", "workers"}}
        if key not in allowed.get(dataset, set()):
            raise SystemExit(f"Unsupported acquisition setting: {left}")
        settings.setdefault(dataset, {})[key] = setting
    if "frames" in settings.get("hypersim", {}):
        if int(settings["hypersim"]["frames"]) < 1 or settings["hypersim"].get("scenes", "all") == "all":
            raise SystemExit("Hypersim smoke frames require a positive count and an explicit scene")
    if "frames" in settings.get("taskonomy", {}):
        values = settings["taskonomy"]
        if int(values["frames"]) < 1 or values.get("num-buildings", "all") == "all":
            raise SystemExit("Taskonomy smoke frames require a positive count and bounded buildings")
    # Scripted gated sources need their input archives, even before they have
    # produced the catalog's expected packed outputs.
    from common.official_sources import missing_manual_assets
    missing = missing_manual_assets(destination, selected)
    if missing:
        for item, directory, details in missing:
            print(f"Missing manual assets: {item.name}\n{item.note}\n"
                  f"Place assets in: {directory}\nRequired: {', '.join(details)}", file=sys.stderr)
        print("After staging these assets, rerun the same command. Independent downloads will continue.", file=sys.stderr)
        available = available_acquisition_items(selected, {item.name for item, _, _ in missing})
        for item in selected:
            if item not in available:
                print(f"SKIP     {item.name}: missing gated inputs or a dependency", file=sys.stderr)
        selected = available
    jobs: list[tuple[str, Path]] = []
    scripted = []
    manual = []
    for item in selected:
        directory = raw_directory(destination, item)
        if item.steps:
            receipt = directory / "sources.json"
            if receipt.is_file():
                old_settings = json.loads(receipt.read_text()).get("settings", {})
                for key in {"frames", "limit", "count"} & old_settings.keys():
                    if old_settings[key] != settings.get(item.name, {}).get(key):
                        raise SystemExit(f"{item.name} contains smoke-only inputs; use a fresh destination for full acquisition")
            official = any(step.script.endswith("acquire_geometry_sources.py") for step in item.steps) or item.name in {"pico_banana", "pd12m"}
            changed = official and (not receipt.is_file() or json.loads(receipt.read_text()).get("settings", {}) != settings.get(item.name, {}))
            if staged_source_status(destination, item)[0] and not changed:
                print(f"EXISTS   {item.name}: {directory}")
            else:
                scripted.append(item)
            continue
        if item.access != "automatic" or not item.files:
            manual.append((item, directory))
            continue
        for url, filename in item.files:
            jobs.append((url, directory / filename))

    for item, directory in manual:
        print(f"\nMANUAL  {item.name}\n  put files in: {directory}\n  {item.note}")
    if args.dry_run:
        for url, path in jobs:
            print(f"DOWNLOAD {url}\n      -> {path}")
        for item in scripted:
            for step in item.steps:
                command, env = step_command(item, step, destination)
                shown = " ".join(f"{key}={value}" for key, value in env.items())
                print(f"SCRIPT   {item.name}: {shown} {' '.join(command)}")
        return 2 if missing else 0

    def fetch(job: tuple[str, Path]) -> Path:
        url, path = job
        if path.is_file() and path.stat().st_size:
            print(f"EXISTS   {path}")
            return path
        print(f"DOWNLOAD {url}\n      -> {path}")
        download_with_resume(url, path)
        return path

    def acquire(item) -> None:
        from common.node_scratch import workspace
        with workspace(f"acquire-{item.name}") as scratch:
            acquire_in_workspace(item, scratch)

    def acquire_in_workspace(item, scratch) -> None:
        from common.node_scratch import acquisition_environment
        directory = raw_directory(destination, item)
        directory.mkdir(parents=True, exist_ok=True)
        for step in item.steps:
            command, env = step_command(item, step, destination)
            if item.name == "pd12m" and settings.get(item.name):
                command = [sys.executable, str(Path(__file__).resolve().parent / "dataset_prep/acquire_geometry_sources.py")]
                env = {"OVB_OFFICIAL_SOURCE": item.name, "OVB_RAW_DIRECTORY": str(directory)}
            process_env = acquisition_environment({**os.environ, **env}, scratch, item.name)
            process_env["OVB_DOWNLOAD_SETTINGS"] = json.dumps(settings.get(item.name, {}))
            # Tools installed next to this interpreter (gdown, hf, ...) must be found
            # even when its environment is not activated.
            process_env["PATH"] = os.pathsep.join(filter(None, (str(Path(sys.executable).parent), process_env.get("PATH"))))
            for attempt in range(1, ATTEMPTS + 1):
                print(f"SCRIPT   {item.name}: {' '.join(command)}", flush=True)
                result = subprocess.run(command, env=process_env, check=False)
                if result.returncode not in NETWORK_EXIT_CODES or attempt == ATTEMPTS:
                    break
                print(f"RETRY    {item.name}: network error (exit {result.returncode}); "
                      f"attempt {attempt + 1}/{ATTEMPTS} in {RETRY_DELAY * attempt}s", flush=True)
                time.sleep(RETRY_DELAY * attempt)
            if result.returncode:
                hint = f"; put the files in {directory}: {item.note}" if item.access == "manual" else ""
                raise RuntimeError(f"{step.script} exited {result.returncode}{hint}")
        (directory / DOWNLOAD_MARKER).touch()

    failures = []
    # Direct files first: scripted sources (COCONut-XL, RefCOCOg, COCO-Object-80) reuse the COCO zips.
    for work, items, workers, label in ((fetch, jobs, args.workers, lambda job: job[0]),
                                        (acquire, scripted, args.script_workers, lambda item: item.name)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(work, entry): entry for entry in items}
            for future in concurrent.futures.as_completed(future_map):
                try:
                    future.result()
                except Exception as exc:  # report every independent download
                    failures.append(f"{label(future_map[future])}: {exc}")
    if failures:
        print("\nFailed downloads:\n  " + "\n  ".join(failures))
        return 1
    if missing:
        print("Public-source acquisition finished; setup remains incomplete until the gated inputs are staged.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    if "--config" in sys.argv[1:] or any(v.startswith("--config=") for v in sys.argv[1:]):
        from scripts.setup_data import main as configured
        raise SystemExit(configured([*sys.argv[1:], "--download-only"]))
    raise SystemExit(main())
