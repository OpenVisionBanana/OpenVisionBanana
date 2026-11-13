"""CPU-only repeat diagnostics; model generation is never performed here."""
from __future__ import annotations

import colorsys
import fcntl
import json
import multiprocessing
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from evaluation.instanceseg.decode import DEFAULT_CONSENSUS_MODE, consensus_vote_threshold, masks_to_coco
from evaluation.instanceseg.postprocess import postprocess_protocol


AGREEMENT_DEFINITION = (
    "Maximum pixel vote count among IoU-matched instance components and background, "
    "divided by all repeats in the prefix. Each repeat contributes at most one vote "
    "per component. Empty predictions vote background; there are no abstentions. "
    "Agreement measures consistency, not correctness."
)
COLOR_SCALES = {
    "grayscale": {"low": "#000000", "high": "#FFFFFF"},
    "diverging": {"low": "#B2182B", "mid": "#F7F7F7", "high": "#2166AC"},
}


def write_json_atomic(path: Path, payload, indent=None) -> None:
    """Prediction dumps stay compact; pass indent for human-read score files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        json.dump(payload, stream, indent=indent)
        temporary = Path(stream.name)
    os.replace(temporary, path)


@contextmanager
def lock_aggregation(out_dir):
    """Prevent two updated merge processes from writing the same run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".aggregation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another aggregation is already writing {out_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def repeat_protocol(args) -> dict:
    """Inference controls stored in every shard, independently of merge arguments."""
    return {
        "version": 2,
        "consensus_mode": args.consensus_mode,
        "model": args.model,
        "transformer_path": args.transformer_path,
        "palette_repeats": args.randomdraw_repeats,
        "generation_repeats": args.generation_seed_repeats,
        "generation_seed": args.generation_seed,
        "sample_seed": args.sample_seed,
        "limit": args.limit,
        "max_instances": args.max_instances,
        "gate": args.gate,
        "gate_file": str(args.gate_file) if args.gate_file else None,
        "prompt_seed": args.prompt_seed if args.prompt_seed is not None else args.sample_seed,
        "prompt_set": args.prompt_set,
        "prompt_template": args.prompt_template,
        "prompt_template_index": args.prompt_template_index,
        "randomize_palette": args.randomize_palette,
        "legend_len": args.legend_len,
        "notation": args.notation,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "batch_size": args.batch_size,
        "native_resolution": args.native_resolution,
        "instance_score": args.instance_score,
        "decode": {"tau": args.tau, "bg_color": list(args.bg_color),
                   "theta_size": args.theta_size, "theta_erosion": args.theta_erosion,
                   "gamma": args.gamma},
        **({"postprocessing": postprocess_protocol(args)} if postprocess_protocol(args) else {}),
        "member_order": "palette_outer_generation_inner; reported repeat numbers are 1-based",
    }


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 0, 0)


def _iou(left, right, area_left, area_right, box_left, box_right):
    x0, y0 = max(box_left[0], box_right[0]), max(box_left[1], box_right[1])
    x1, y1 = min(box_left[2], box_right[2]), min(box_left[3], box_right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = np.count_nonzero(left[y0:y1, x0:x1] & right[y0:y1, x0:x1])
    return intersection / (area_left + area_right - intersection) if intersection else 0.0


def normalized_repeat_protocol(protocol):
    """Legacy v1 predictions used strict majority; never relabel them as inclusive."""
    if protocol is None:
        return None
    protocol = dict(protocol)
    if protocol.get("version") == 1:
        protocol.setdefault("consensus_mode", "strict_majority")
        protocol["version"] = 2
    return protocol


class CumulativeInstanceConsensus:
    """Incrementally build the same cross-repeat IoU graph as consensus_instance_masks.

    New members add edges; every snapshot recomputes components and votes under
    the selected consensus mode for its own prefix. Later repeats cannot influence an
    earlier prefix's matching, even when they bridge previously separate objects.
    """

    def __init__(self, shape, match_iou=0.5, consensus_mode=DEFAULT_CONSENSUS_MODE):
        if not 0 < match_iou <= 1:
            raise ValueError("match_iou must be in (0, 1]")
        self.shape = tuple(shape)
        self.match_iou = match_iou
        consensus_vote_threshold(1, consensus_mode)  # Validate even before the first snapshot.
        self.consensus_mode = consensus_mode
        self.nodes = []
        self.parents = []
        self.count = 0
        self.background_votes = np.zeros(self.shape, dtype=np.uint32)
        self.first_masks = []

    def _find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def add(self, masks):
        foreground = np.zeros(self.shape, dtype=bool)
        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.shape:
                raise ValueError(f"Mismatched mask shape {mask.shape}, expected {self.shape}")
            if self.count == 0:
                self.first_masks.append(mask)
            foreground |= mask
            area, box = int(mask.sum()), _bbox(mask)
            index = len(self.nodes)
            self.parents.append(index)
            for previous, (repeat, other, other_area, other_box) in enumerate(self.nodes):
                if repeat == self.count or not area or not other_area:
                    continue
                if _iou(mask, other, area, other_area, box, other_box) >= self.match_iou:
                    self.parents[self._find(index)] = self._find(previous)
            self.nodes.append((self.count, mask, area, box))
        self.background_votes += ~foreground
        self.count += 1

    def snapshot(self, diagnostics=False):
        if not self.count:
            raise ValueError("No repeat has been added")
        groups = defaultdict(list)
        for index, node in enumerate(self.nodes):
            groups[self._find(index)].append(node)
        required = consensus_vote_threshold(self.count, self.consensus_mode)
        output, objects = [], []
        agreement_votes = self.background_votes.copy() if diagnostics else None
        foreground_votes = np.zeros(self.shape, dtype=np.uint32) if diagnostics else None
        for nodes in groups.values():
            by_repeat = defaultdict(list)
            for repeat, mask, _area, _box in nodes:
                by_repeat[repeat].append(mask)
            if not diagnostics and len(by_repeat) < required:
                continue
            votes = np.zeros(self.shape, dtype=np.uint32)
            for masks in by_repeat.values():
                votes += np.logical_or.reduce(masks)
            consensus = votes >= required
            kept = len(by_repeat) >= required and bool(consensus.any())
            if kept:
                output.append(consensus)
            if diagnostics:
                np.maximum(agreement_votes, votes, out=agreement_votes)
                np.maximum(foreground_votes, votes, out=foreground_votes)
                objects.append({"support": len(by_repeat), "repeats": self.count,
                                "kept": kept, "bbox_xyxy": list(_bbox(votes > 0))})
        # The one-pass API preserves mask order (and even empty masks) exactly.
        output = self.first_masks if self.count == 1 else sorted(output, key=lambda mask: int(mask.sum()), reverse=True)
        if not diagnostics:
            return output, None
        return output, {"agreement": agreement_votes.astype(np.float32) / self.count,
                        "instance_support": foreground_votes.astype(np.float32) / self.count,
                        "objects": objects}


@lru_cache(maxsize=8)
def _font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def colorize_agreement(values, count, title="Matched-instance/background vote agreement"):
    # Share the actual semseg conversion, not an approximation of its palette.
    from evaluation.semseg.inference_partial_flux2_klein_img2img import vote_agreement_to_rgb

    colorized = Image.fromarray(vote_agreement_to_rgb(values))
    width = max(640, colorized.width)
    canvas = Image.new("RGB", (width, colorized.height + 100), "white")
    canvas.paste(colorized, ((width - colorized.width) // 2, 0))
    draw, font = ImageDraw.Draw(canvas), _font()
    draw.text((16, colorized.height + 6), f"{title} | {count} repeats", fill="black", font=font)
    gradient = np.linspace(0, 1, width - 48, dtype=np.float32)[None, :]
    bar = Image.fromarray(vote_agreement_to_rgb(np.repeat(gradient, 12, axis=0)))
    canvas.paste(bar, (24, colorized.height + 30))
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = 24 + round(fraction * (width - 49))
        draw.text((x, colorized.height + 47), f"{fraction:.0%}", fill="black", font=font,
                  anchor="la" if fraction == 0 else "ra" if fraction == 1 else "ma")
    draw.text((16, colorized.height + 73), "Consistency, not correctness. Empty masks vote background.", fill="black", font=font)
    return canvas


class DisplayInstanceIds:
    """One-to-one spatial matching for display only; never changes scoring masks."""

    def __init__(self):
        self.masks = {}
        self.next_id = 1

    def assign(self, masks):
        identities = [None] * len(masks)
        old_ids = list(self.masks)
        if masks and old_ids:
            overlaps = np.zeros((len(masks), len(old_ids)))
            for row, mask in enumerate(masks):
                for col, identity in enumerate(old_ids):
                    other = self.masks[identity]
                    overlaps[row, col] = _iou(mask, other, int(mask.sum()), int(other.sum()), _bbox(mask), _bbox(other))
            rows, cols = linear_sum_assignment(-overlaps)
            for row, col in zip(rows, cols):
                if overlaps[row, col] >= 0.3:
                    identities[row] = old_ids[col]
        for index, mask in enumerate(masks):
            if identities[index] is None:
                identities[index] = self.next_id
                self.next_id += 1
            self.masks[identities[index]] = mask
        return identities


def _fit_mask(mask, shape):
    """Rescale a GT mask that was encoded at another resolution than its image.

    Fourteen SA-Co/Gold images carry GT RLEs whose size differs from the images[]
    entry (same aspect ratio; every annotator part agrees, e.g. metaclip
    420000000 at 596x900 against a 795x1200 image). Panels draw GT over the image
    itself, so rescale rather than drop those objects.
    """
    if mask.shape == tuple(shape):
        return mask
    height, width = shape
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
    return np.asarray(resized) > 127


def _instance_overlay(image, masks, identities):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, identity in zip(masks, identities):
        color = np.array(colorsys.hsv_to_rgb((identity * 0.61803398875) % 1, 0.8, 1)) * 255
        arr[mask] = arr[mask] * 0.3 + color * 0.7
    result = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for mask, identity in zip(masks, identities):
        x0, y0, _x1, _y1 = _bbox(mask)
        if mask.any():
            draw.text((x0, y0), str(identity), font=_font(), fill="white", stroke_width=1, stroke_fill="black")
    return result


def foreground_change_overlay(image, previous, current):
    shape = (image.height, image.width)
    before = np.logical_or.reduce(previous) if previous else np.zeros(shape, dtype=bool)
    after = np.logical_or.reduce(current) if current else np.zeros(shape, dtype=bool)
    arr = np.asarray(image.convert("L").convert("RGB")).copy()
    arr[after & ~before] = (0, 200, 90)
    arr[before & ~after] = (220, 0, 180)
    return Image.fromarray(arr)


def _grid(frames, path, title):
    width, height, caption, columns = 320, 240, 30, 4
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, 42 + rows * (height + caption)), "white")
    draw, font = ImageDraw.Draw(canvas), _font()
    draw.text((10, 10), title, fill="black", font=font)
    for index, (label, image) in enumerate(frames):
        cell = image.copy()
        cell.thumbnail((width, height), Image.Resampling.NEAREST)
        x, y = index % columns * width, 42 + index // columns * (height + caption)
        canvas.paste(cell, (x + (width - cell.width) // 2, y + (height - cell.height) // 2))
        draw.text((x + 6, y + height + 5), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


class RepeatVisualizations:
    def __init__(self, root, image_id, image, gt_masks, total, noun_phrase):
        self.root, self.image_id, self.image, self.total = root, image_id, image, total
        self.noun_phrase = noun_phrase
        self.display_ids = DisplayInstanceIds()
        gt_ids = self.display_ids.assign(gt_masks)
        self.frames = [("Input", image), (f"GT annotator a: {len(gt_masks)} objects", _instance_overlay(image, gt_masks, gt_ids))]
        self.changes = []
        self.previous = []
        self.summary = []

    def add(self, count, masks, diagnostics):
        prefix = f"1to{count:02d}"
        ids = self.display_ids.assign(masks)
        self.frames.append((f"{prefix}: {len(masks)} objects", _instance_overlay(self.image, masks, ids)))
        for directory, rendered in (
            ("vote_agreement", Image.fromarray((diagnostics["agreement"] * 255).astype(np.uint8))),
            ("vote_agreement_color", colorize_agreement(diagnostics["agreement"], count)),
        ):
            target = self.root / directory / prefix / f"{self.image_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(target)
            if count == self.total:
                rendered.save(self.root / directory / f"{self.image_id}.png")
        support = colorize_agreement(diagnostics["instance_support"], count, "Foreground instance support (background = 0)")
        draw = ImageDraw.Draw(support)
        x_offset = (support.width - self.image.width) // 2
        for obj in diagnostics["objects"]:
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            if x1 <= x0 or y1 <= y0:
                continue
            label = f"{obj['support']}/{count} {'kept' if obj['kept'] else 'dropped'}"
            draw.text((x_offset + x0, y0), label, fill="white", font=_font(), stroke_width=1, stroke_fill="black")
        support_path = self.root / "instance_support" / prefix / f"{self.image_id}.png"
        support_path.parent.mkdir(parents=True, exist_ok=True)
        support.save(support_path)
        if count > 1:
            change = foreground_change_overlay(self.image, self.previous, masks)
            change_path = self.root / "repeat_changes" / prefix / f"{self.image_id}.png"
            change_path.parent.mkdir(parents=True, exist_ok=True)
            change.save(change_path)
            self.changes.append((f"1to{count - 1:02d} -> {prefix}", change))
        self.previous = masks
        self.summary.append({"prefix": count, "num_instances": len(masks), "display_ids": ids,
                             "objects": diagnostics["objects"]})

    def finish(self):
        _grid(self.frames, self.root / "repeat_progression" / f"{self.image_id}.png",
              f"{self.noun_phrase} | matched display colors; GT=a (scores use oracle a/b/c)")
        if self.changes:
            _grid(self.changes, self.root / "repeat_changes" / f"{self.image_id}.png",
                  "Foreground changes: green=added, magenta=removed; neither implies correctness")
        write_json_atomic(self.root / "repeat_progression" / f"{self.image_id}.json",
                          {"agreement_definition": AGREEMENT_DEFINITION, "prefixes": self.summary})


def validate_repeat_shard(data, args, expected_samples):
    """Verify aggregation mode as well as inference settings and query coverage."""
    fields = ("repeat_protocol", "repeat_preds", "repeat_queries")
    if not any(field in data for field in fields):
        saved_mode = data.get("consensus_mode", "strict_majority")
        if args.randomdraw_repeats * args.generation_seed_repeats > 1 and args.consensus_mode != saved_mode:
            label = "Legacy final-only" if "consensus_mode" not in data else "Final-only"
            raise ValueError(f"{label} shards used {saved_mode}; merge them with "
                             f"--consensus_mode {saved_mode}. Re-vote saved repeat masks "
                             "separately to change the consensus mode.")
        return False
    if not all(field in data for field in fields):
        raise ValueError(f"Shard {data.get('rank')} has incomplete repeat data")
    saved = normalized_repeat_protocol(data["repeat_protocol"])
    if saved.get("consensus_mode") != args.consensus_mode:
        raise ValueError(f"Shard {data.get('rank')} consensus mode differs: "
                         f"saved={saved.get('consensus_mode')!r}, requested={args.consensus_mode!r}. "
                         "Use --consensus_mode with the saved mode, or re-vote saved repeat masks separately.")
    if saved != repeat_protocol(args):
        raise ValueError(f"Shard {data.get('rank')} repeat protocol differs from the aggregation settings")
    repeats = args.randomdraw_repeats * args.generation_seed_repeats
    if not isinstance(data["repeat_preds"], list) or len(data["repeat_preds"]) != repeats:
        raise ValueError(f"Shard {data.get('rank')} must contain exactly {repeats} repeat prediction dictionaries")
    expected = {(sample.subset, sample.image_id, sample.np_text) for sample in expected_samples}
    queries = data["repeat_queries"]
    actual = {(query["subset"], query["image_id"], query["np_text"]) for query in queries}
    if actual != expected or len(queries) != len(expected_samples) or data["num_samples"] != len(expected_samples):
        raise ValueError(f"Shard {data.get('rank')} repeat query coverage differs from the selected sample")
    if any(type(query["positive"]) is not bool for query in queries):
        raise ValueError("Repeat query gate decisions must be booleans")
    if sum(not query["positive"] for query in queries) != data["num_gated_out"]:
        raise ValueError("Repeat query gate counts differ from the shard totals")
    positive = {(query["subset"], query["image_id"]) for query in queries if query["positive"]}
    for member in data["repeat_preds"]:
        if not isinstance(member, dict):
            raise ValueError("Repeat predictions must be grouped by subset")
        for subset, predictions in member.items():
            if any((subset, prediction["image_id"]) not in positive for prediction in predictions):
                raise ValueError("Repeat predictions contain an unselected or gated-out query")
    return True


def analysis_worker_count():
    """Bound CPU workers by both the allocation and this process's affinity."""
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    requested = int(os.environ.get("REPEAT_ANALYSIS_WORKERS", str(min(8, allocated))))
    if requested < 1 or allocated < 1:
        raise ValueError("REPEAT_ANALYSIS_WORKERS and SLURM_CPUS_PER_TASK must be positive")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # Outside Slurm, an explicit request may use the local CPU affinity.
    limit = min(available, allocated) if "SLURM_CPUS_PER_TASK" in os.environ else available
    return min(requested, limit)


@contextmanager
def _analysis_pool(workers):
    if workers == 1:
        yield None
        return
    # Spawn, never fork an evaluator that already imported torch/BLAS. Limit
    # numerical libraries before child imports, avoiding workers x BLAS threads.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            yield pool
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _resume_scores(summary_path, protocol, repeats, run_eval):
    """Reuse a completed score pass so a failed visualization run costs only itself.

    Set REPEAT_ANALYSIS_RESUME=1 to skip standalone and cumulative scoring when
    this directory already holds every score for this exact protocol. Predictions
    stay on disk from that pass; nothing else is reused.
    """
    if os.environ.get("REPEAT_ANALYSIS_RESUME", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if not run_eval or not summary_path.exists():
        return None
    previous = json.loads(summary_path.read_text())
    if normalized_repeat_protocol(previous.get("repeat_protocol")) != protocol or previous.get("scores_status") != "complete":
        return None
    if len(previous.get("standalone", [])) != repeats or len(previous.get("cumulative", [])) != repeats:
        return None
    if any(entry.get("scores") is None for entry in previous["standalone"] + previous["cumulative"]):
        return None
    return previous


def _repeat_query(task):
    """Independent query worker; only compact RLEs cross process boundaries."""
    sample, members, score, decode_rle_fn, visual_args, consensus_mode = task
    shape = (sample.gt_h, sample.gt_w)
    state = CumulativeInstanceConsensus(shape, consensus_mode=consensus_mode)
    visual = None
    if visual_args is not None:
        root, ground_truth = visual_args
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB").resize((sample.gt_w, sample.gt_h), Image.Resampling.LANCZOS)
        masks = [_fit_mask(decode_rle_fn(rle, *shape), shape) for rle in ground_truth]
        visual = RepeatVisualizations(root, sample.image_id, image, masks, len(members), sample.np_text)
    cumulative = []
    for index, predictions in enumerate(members):
        masks = [decode_rle_fn(pred["segmentation"], *shape) for pred in predictions]
        state.add(masks)
        voted, diagnostics = state.snapshot(diagnostics=visual is not None)
        if visual is None:
            cumulative.append(masks_to_coco(voted, sample.image_id, score))
        else:
            visual.add(index + 1, voted, diagnostics)
    if visual is not None:
        visual.finish()
    return cumulative


def _query_results(pool, tasks, workers, heartbeat):
    """Bound in-flight work, report completions even if an earlier query is slow."""
    if pool is None:
        for index, task in enumerate(tasks):
            yield index, _repeat_query(task)
        return
    tasks = iter(enumerate(tasks))
    pending = {}

    def submit_next():
        item = next(tasks, None)
        if item is not None:
            index, task = item
            pending[pool.submit(_repeat_query, task)] = index

    for _ in range(workers * 2):
        submit_next()
    while pending:
        done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
        if not done:
            heartbeat()
        for future in done:
            index = pending.pop(future)
            yield index, future.result()
            submit_next()


def analyze_repeat_outputs(**kwargs):
    try:
        return _analyze_repeat_outputs(**kwargs)
    except (Exception, KeyboardInterrupt) as exc:
        path = kwargs["out_dir"] / "repeat_analysis_results.json"
        if path.exists():
            summary = json.loads(path.read_text())
            summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           error=f"{type(exc).__name__}: {exc}")
            write_json_atomic(path, summary, indent=2)
        raise


def _analyze_repeat_outputs(*, out_dir, gt_folder, subsets, shard_paths, all_samples,
                           args, run_eval_fn, load_gt_masks_fn, decode_rle_fn, final_results):
    """Score standalone members first, parallelize prefixes, then render visuals.

    Only the parent writes prediction/score JSONs. Progress and completed scores
    are published atomically throughout; a partial summary is never 'complete'.
    """
    repeats = args.randomdraw_repeats * args.generation_seed_repeats
    workers = analysis_worker_count()
    protocol = repeat_protocol(args)
    summary_path = out_dir / "repeat_analysis_results.json"
    summary = {
        "status": "running", "phase": "loading_shards", "scores_status": "pending",
        "repeat_protocol": protocol,
        "num_queries": len(all_samples), "num_subsets": len(subsets),
        "agreement_definition": AGREEMENT_DEFINITION,
        "vote_agreement_color_scales": COLOR_SCALES,
        "visualization_queries_per_subset": {},
        "standalone": [], "cumulative": [],
        "progress": {"workers": workers, "completed_prefix_subsets": []},
    }

    def publish(phase=None):
        if phase is not None:
            summary["phase"] = phase
        write_json_atomic(summary_path, summary, indent=2)

    resumed = _resume_scores(summary_path, protocol, repeats, args.run_eval)
    publish()
    if resumed is None:
        print(f"Repeat analysis: {workers} CPU worker(s); standalone scores first, "
              "then cumulative scores, then optional visualizations.", flush=True)
    else:
        print(f"Repeat analysis: {workers} CPU worker(s); REPEAT_ANALYSIS_RESUME keeps the "
              f"{repeats} standalone and {repeats} cumulative scores already in this directory, "
              "rendering visualizations only.", flush=True)
    raw = [defaultdict(list) for _ in range(repeats)]
    positive = set()
    for path in shard_paths:
        payload = json.loads(path.read_text())
        # A resumed run rescores nothing, so the member predictions it would
        # rewrite are never held in memory.
        for index, member in enumerate(payload["repeat_preds"] if resumed is None else []):
            for subset, predictions in member.items():
                raw[index][subset].extend(predictions)
        positive.update((query["subset"], query["image_id"]) for query in payload["repeat_queries"] if query["positive"])
    # Do not keep the final shard payload alive while processing the full run.
    del payload

    by_subset, eval_ids = defaultdict(list), defaultdict(set)
    panel_ids = defaultdict(set)
    positive_counts = defaultdict(int)
    for sample in all_samples:
        by_subset[sample.subset].append(sample)
        eval_ids[sample.subset].add(sample.image_id)
        if args.save_panels and (sample.subset, sample.image_id) in positive:
            positive_counts[sample.subset] += 1
            if positive_counts[sample.subset] == 1 or positive_counts[sample.subset] % args.panel_every == 0:
                panel_ids[sample.subset].add(sample.image_id)

    summary["visualization_queries_per_subset"] = {subset: len(panel_ids[subset]) for subset in subsets}
    summary["visualizations_status"] = "pending" if args.save_panels else "disabled"

    def score(pred_root, result_path, aggregation, members, cached=None):
        provenance = {"gate": args.gate, "gate_file": str(args.gate_file) if args.gate_file else None,
                      "batch_size": args.batch_size, "repeat_protocol": protocol,
                      "consensus_mode": args.consensus_mode,
                      "aggregation": aggregation, "members": members}
        if not args.run_eval:
            return None
        result = ({**cached, **provenance} if cached is not None else
                  run_eval_fn(pred_root=pred_root, gt_folder=gt_folder, subsets=subsets,
                              iou_type=args.iou_type, verbose=False, out_json=None,
                              eval_ids_by_subset=eval_ids, provenance=provenance))
        if (result.get("aggregate") or {}).get("num_subsets") != len(subsets):
            raise RuntimeError("Repeat scoring did not evaluate every requested subset")
        # Same layout as saco_gold_results.json (eval_saco_gold writes indent=2).
        write_json_atomic(result_path, result, indent=2)
        return result

    if resumed is not None:
        summary.update(standalone=resumed["standalone"], cumulative=resumed["cumulative"],
                       scores_status=resumed["scores_status"])
        summary["progress"].update(standalone_completed=repeats, cumulative_completed=repeats,
                                   completed_prefix_subsets=list(subsets))

    publish("standalone_scores")
    first_result = None
    for index in range(1, repeats + 1) if resumed is None else ():
        palette_index, generation_index = divmod(index - 1, args.generation_seed_repeats)
        members = [{"repeat": index, "palette_repeat": palette_index + 1,
                    "generation_repeat": generation_index + 1}]
        prediction_root = out_dir / "repeat_predictions" / f"repeat{index:02d}"
        for subset in subsets:
            write_json_atomic(prediction_root / subset / "coco_predictions_segm.json", raw[index - 1][subset])
        result_path = out_dir / f"saco_gold_results_repeat{index:02d}.json"
        print(f"Repeat analysis: standalone {index}/{repeats} (all {len(subsets)} subsets).", flush=True)
        result = score(prediction_root, result_path, "standalone_repeat", members)
        if index == 1:
            first_result = result
        summary["standalone"].append({"repeat": index, "members": members,
                                      "predictions": str(prediction_root.relative_to(out_dir)),
                                      "result_file": result_path.name if result is not None else None,
                                      "scores": result})
        summary["progress"]["standalone_completed"] = index
        publish()

    # Raw RLEs are now persisted; release them and load one subset at a time.
    del raw

    def process_subset(subset, pool, visuals=False):
        per_repeat = []
        for index in range(repeats):
            path = out_dir / "repeat_predictions" / f"repeat{index + 1:02d}" / subset / "coco_predictions_segm.json"
            grouped = defaultdict(list)
            for pred in json.loads(path.read_text()):
                grouped[pred["image_id"]].append(pred)
            per_repeat.append(grouped)
        selected = [sample for sample in by_subset[subset]
                    if (sample.image_id in panel_ids[subset] if visuals else
                        (subset, sample.image_id) in positive)]
        gt_masks = load_gt_masks_fn(subset, gt_folder, panel_ids[subset]) if visuals and selected else {}
        tasks = ((sample, [member.get(sample.image_id, []) for member in per_repeat],
                  args.instance_score, decode_rle_fn,
                  (out_dir / subset, gt_masks.get(sample.image_id, [])) if visuals else None,
                  args.consensus_mode)
                 for sample in selected)
        started = last_report = time.monotonic()
        completed = 0
        results = {}

        def report(force=False):
            nonlocal last_report
            now = time.monotonic()
            if force or now - last_report >= 15:
                elapsed = now - started
                rate = completed / elapsed if elapsed else 0
                summary["progress"].update(subset=subset, queries_completed=completed,
                                           queries_total=len(selected), elapsed_seconds=round(elapsed, 1),
                                           queries_per_second=round(rate, 3))
                publish()
                print(f"Repeat analysis [{summary['phase']}]: {subset} {completed}/{len(selected)} "
                      f"positive queries; {rate:.2f} queries/s; {workers} worker(s).", flush=True)
                last_report = now

        report(force=True)
        for index, predictions in _query_results(pool, tasks, workers, report):
            if not visuals:
                results[index] = predictions
            completed += 1
            report()
        report(force=True)
        if visuals:
            return
        cumulative = [[] for _ in range(repeats)]
        # Completion order never changes prediction order or score tie-breaking.
        for index in range(len(selected)):
            for prefix, predictions in enumerate(results.pop(index)):
                cumulative[prefix].extend(predictions)
        canonical = json.loads((out_dir / subset / "coco_predictions_segm.json").read_text())
        canonicalize = lambda preds: sorted(json.dumps(pred, sort_keys=True) for pred in preds)
        if canonicalize(canonical) != canonicalize(cumulative[-1]):
            raise RuntimeError(f"{subset}: final cumulative masks differ from saved consensus")
        for index, predictions in enumerate(cumulative, start=1):
            write_json_atomic(out_dir / "cumulative_predictions" / f"1to{index:02d}" / subset / "coco_predictions_segm.json", predictions)
        summary["progress"]["completed_prefix_subsets"].append(subset)
        publish()

    if resumed is None:
        publish("cumulative_predictions")
        with _analysis_pool(workers) as pool:
            for subset in subsets:
                process_subset(subset, pool)

    publish("cumulative_scores")
    for count in range(1, repeats + 1) if resumed is None else ():
        prediction_root = out_dir / "cumulative_predictions" / f"1to{count:02d}"
        result_path = out_dir / f"saco_gold_results_1to{count:02d}.json"
        members = [entry["members"][0] for entry in summary["standalone"][:count]]
        print(f"Repeat analysis: cumulative 1to{count:02d}/{repeats} (all {len(subsets)} subsets).", flush=True)
        result = score(prediction_root, result_path, "cumulative_instance_consensus", members,
                       cached=first_result if count == 1 else final_results if count == repeats else None)
        summary["cumulative"].append({"first_repeat": 1, "last_repeat": count,
                                      "predictions": str(prediction_root.relative_to(out_dir)),
                                      "result_file": result_path.name if result is not None else None,
                                      "scores": result})
        summary["progress"]["cumulative_completed"] = count
        publish()
    if resumed is None:
        summary["scores_status"] = "complete" if args.run_eval else "disabled"
    if args.run_eval and len(summary["standalone"]) == repeats:
        summary["mean_per_pass"] = {
            key: round(sum(entry["scores"]["aggregate"][key] for entry in summary["standalone"]) / repeats, 4)
            for key in ("cgF1", "pmF1", "IL_MCC")
        }
    publish()
    if args.save_panels:
        # Heatmaps cannot delay or hold back any score JSON. Use the same masks,
        # matching, selection interval, and semantic-segmentation color scales.
        publish("visualizations")
        with _analysis_pool(workers) as pool:
            for subset in subsets:
                process_subset(subset, pool, visuals=True)
        summary["visualizations_status"] = "complete"
    summary["status"] = "complete" if args.run_eval else "predictions_only"
    publish("complete")
    return summary["status"]
