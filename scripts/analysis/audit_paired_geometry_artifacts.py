#!/usr/bin/env python3
"""Diagnostic paired audit; never modifies predictions or benchmark scores."""
import argparse
import csv
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PAIRS = {
    '4b_individual': ('4b_zeroshot_checkpoint87500_95000_w25', '4b_nonzero_checkpoint100000'),
    '9b_individual': ('9b_zeroshot_checkpoint100000', '9b_nonzero_checkpoint75000'),
    '4b_released': ('4b_zeroshot_checkpoint87500_95000_w25', '4b_nonzero_avg85000_100000'),
    '9b_released': ('9b_zeroshot_coco86750_w25', '9b_nonzero_avg75000_100000_drop77500_82500'),
}


def records(folder):
    paths = list(folder.glob('*_per_sample.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'Expected one score file: {folder}')
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    result = {r['sample_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate sample IDs')
    return result


def check_generation(folders):
    documents = [json.loads(next(f.glob('*_results.json')).read_text()) for f in folders]
    keys = ('model_family', 'prompt_sha256', 'steps', 'guidance_scale', 'seed', 'seed_mode', 'native_resolution')
    for key in keys:
        a, b = [d.get('generation', {}).get(key) for d in documents]
        if a != b or (a is None and key in ('prompt_sha256', 'steps', 'guidance_scale')):
            raise ValueError(f'Incompatible or missing generation {key}: {a!r}, {b!r}')


def scene(row):
    path = row['image_path']
    for pattern in (r'(scene\d{4}_\d{2})', r'(Scene\d+)', r'(\d{4}_\d{2}_\d{2}_drive_\d+_sync)',
                    r'((?:indoors|outdoor)/scene_\d+)', r'/images/([^/]+)/', r'/test/([^/]+)/'):
        match = re.search(pattern, path)
        if match: return match.group(1)
    return row['sample_id'].split('_')[0]


class Predictions:
    def __init__(self, folder):
        self.tar = tarfile.open(folder / 'pred_rgb_raw.tar')
        self.members = {Path(m.name).stem: m for m in self.tar.getmembers() if m.isfile() and m.name.endswith('.png')}
    def image(self, stem):
        with self.tar.extractfile(self.members[stem]) as f:
            return Image.open(io.BytesIO(f.read())).convert('RGB')
    def close(self):
        self.tar.close()


def normal_artifacts(rgb, gt, mask):
    from common.surface_normal_encoding import decode_surface_normals_rgb
    pred, valid = decode_surface_normals_rgb(rgb, encoding='vision_banana_x_left_red')
    angle = np.degrees(np.arccos(np.clip(np.sum(pred * gt, axis=-1), -1, 1)))
    angle[~valid] = 90
    # Near-neutral is a fill-like candidate, NOT automatically an invalid prediction.
    neutral = np.linalg.norm(rgb.astype(np.float32) / 127.5 - 1, axis=-1) < 0.05
    smooth_count = 0
    excess_sum = 0.0
    rough_count = 0
    for axis in (0, 1):
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        g = np.degrees(np.arccos(np.clip(np.sum(gt[a] * gt[b], axis=-1), -1, 1)))
        p = np.degrees(np.arccos(np.clip(np.sum(pred[a] * pred[b], axis=-1), -1, 1)))
        use = mask[a] & mask[b] & valid[a] & valid[b] & (g < 2)
        smooth_count += int(use.sum())
        excess_sum += float(np.maximum(p[use] - g[use], 0).sum())
        rough_count += int((p[use] > 10).sum())
    n = int(mask.sum())
    if n == 0:
        raise ValueError('No GT-valid pixels')
    return {'near_neutral_fraction': float(neutral[mask].mean()),
            'invalid_fraction': float((~valid)[mask].mean()),
            'angle_over_30_fraction': float((angle[mask] > 30).mean()),
            'smooth_surface_excess_deg': excess_sum / smooth_count if smooth_count else None,
            'smooth_surface_rough_fraction': rough_count / smooth_count if smooth_count else None,
            'smooth_pairs': smooth_count}, angle


def write_report(output, rows, metadata):
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for pair, dataset in sorted({(r['pair'], r['dataset']) for r in rows}):
        subset = [r for r in rows if r['pair'] == pair and r['dataset'] == dataset]
        groups.append({'pair': pair, 'dataset': dataset, 'images': len(subset),
                       'mean_error_delta': float(np.mean([r['error_delta'] for r in subset])),
                       'fraction_nonzero_worse': float(np.mean([r['error_delta'] > 0 for r in subset])),
                       'top_regressions': sorted(subset, key=lambda r: r['error_delta'], reverse=True)[:20],
                       'top_improvements': sorted(subset, key=lambda r: r['error_delta'])[:10],
                       'artifact_means': {k: float(np.mean([r[k] for r in subset if r.get(k) is not None]))
                                          for k in keys if k.startswith(('zs_', 'nzs_')) and any(r.get(k) is not None for r in subset)}})
        groups[-1]['scene_groups'] = [{'scene': s, 'images': len(v), 'mean_error_delta': float(np.mean([r['error_delta'] for r in v]))}
                                     for s in sorted({r['scene'] for r in subset})
                                     for v in [[r for r in subset if r['scene'] == s]]]
    output.with_suffix('.json').write_text(json.dumps({'status': 'complete', 'metadata': metadata, 'groups': groups}, indent=2, allow_nan=False))
    lines = ['# Paired geometry diagnostic audit', '', '| Pair | Dataset | Images | NZS minus ZS error | NZS worse |', '|---|---|---:|---:|---:|']
    lines += [f'| {g["pair"]} | {g["dataset"]} | {g["images"]} | {g["mean_error_delta"]:+.4f} | {100*g["fraction_nonzero_worse"]:.1f}% |' for g in groups]
    lines += ['', 'Positive differences indicate non-zero-shot regression. ' + metadata['limits']]
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f'PASS: {len(rows)} paired records -> {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    if args.dataset == 'depth':
        for pair, (zs, nzs) in PAIRS.items():
            for dataset in ('nyu', 'kitti_eigen', 'eth3d', 'scannet', 'diode'):
                folders = [args.results_root / 'depth_marigold_v2_ppd' / label / dataset for label in (zs, nzs)]
                check_generation(folders)
                a, b = map(records, folders)
                if set(a) != set(b): raise ValueError('Unpaired depth samples')
                for sid in sorted(a):
                    metric = 'marigold_v2_ppd_abs_rel'
                    rows.append({'pair': pair, 'dataset': dataset, 'sample_id': sid, 'scene': scene(a[sid]),
                                 'image_path': a[sid]['image_path'], 'zs_error': a[sid][metric],
                                 'nzs_error': b[sid][metric], 'error_delta': b[sid][metric] - a[sid][metric]})
        write_report(args.output, rows, {'metric': 'corrected cached GT-scored PPD AbsRel',
            'limits': 'Depth pixel artifacts and cross-task masks not yet measured; no black-pixel invalidity assumptions.'})
        return
    from common.surface_normal_encoding import transform_surface_normals, encode_surface_normals_rgb
    from evaluation.surface_normals import datasets
    from evaluation.surface_normals.inference_partial_flux2_klein_img2img import load_gt_normals, load_mask, resize_rgb
    from evaluation.surface_normals.target_protocol import resolve_ground_truth_protocol
    root = datasets.resolve_dataset_root(args.dataset, args.data_root)
    cases = ('indoor',) if args.dataset == 'diode' else ('standard',)
    samples = {s.sample_id: s for s in datasets.build_samples(args.dataset, root, 'test', None, cases, ('appearance',))}
    info_path = root / 'dataset_info.json'
    if not info_path.exists(): info_path = root / 'manifest.json'
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    protocol = resolve_ground_truth_protocol(dataset_info=info, storage_formats=sorted({s.normal_path.suffix for s in samples.values()}))
    encoding = 'legacy_direct_xyz' if protocol.encoding.startswith('float_xyz') else protocol.encoding
    for pair, (zs, nzs) in PAIRS.items():
        folders = [args.results_root / 'surface_normals_geometry_v6' / label / args.dataset for label in (zs, nzs)]
        check_generation(folders)
        a, b = map(records, folders)
        if set(a) != set(b): raise ValueError('Unpaired normal samples')
        ranked = sorted(a, key=lambda sid: b[sid]['mean_angle'] - a[sid]['mean_angle'], reverse=True)
        chosen = set(ranked[:4] + ranked[-2:] + ranked[len(ranked)//2:len(ranked)//2+2])
        archives = [Predictions(f) for f in folders]
        gallery = []
        try:
            for i, sid in enumerate(ranked):
                sample = samples[sid]
                gt, gv = load_gt_normals(sample.normal_path, min_norm=0.01, rgb_encoding=encoding)
                gt = transform_surface_normals(gt, protocol.permutation, protocol.signs)
                mask = load_mask(sample.mask_path, gt.shape[:2], gv) & gv
                imgs = [ar.image(sample.output_stem) for ar in archives]
                measurements = [normal_artifacts(np.asarray(resize_rgb(im, gt.shape[:2])), gt, mask) for im in imgs]
                row = {'pair': pair, 'dataset': args.dataset, 'sample_id': sid, 'scene': scene(a[sid]), 'image_path': a[sid]['image_path'],
                       'zs_error': a[sid]['mean_angle'], 'nzs_error': b[sid]['mean_angle'],
                       'error_delta': b[sid]['mean_angle'] - a[sid]['mean_angle'], 'gt_valid_pixels': int(mask.sum())}
                for prefix, (metrics, _) in zip(('zs_', 'nzs_'), measurements):
                    row.update({prefix + k: v for k, v in metrics.items()})
                rows.append(row)
                if sid in chosen:
                    panels = [Image.open(sample.image_path).convert('RGB'),
                              Image.fromarray(encode_surface_normals_rgb(gt, mask, encoding='vision_banana_x_left_red')), *imgs]
                    for _, angle in measurements:
                        heat = np.zeros((*mask.shape, 3), dtype=np.uint8)
                        heat[..., 0] = np.clip(angle / 90 * 255, 0, 255).astype('uint8')
                        heat[..., 2] = 255 - heat[..., 0]; heat[~mask] = 0
                        panels.append(Image.fromarray(heat))
                    canvas = Image.new('RGB', (6 * 240, 208), 'white')
                    draw = ImageDraw.Draw(canvas)
                    for j, (im, label) in enumerate(zip(panels, ('Input', 'GT', 'ZS', 'NZS', 'ZS error 0-90deg', 'NZS error 0-90deg'))):
                        im.thumbnail((240, 180)); canvas.paste(im, (j * 240, 28))
                        draw.text((j * 240, 0), label, fill='black')
                    draw.text((0, 14), f'{sid} delta={row["error_delta"]:+.2f} deg', fill='black')
                    gallery.append(canvas)
                if i % 50 == 0: print(f'{pair}/{args.dataset} {i}/{len(ranked)}', flush=True)
        finally:
            for ar in archives: ar.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (1440, len(gallery)*208), 'white')
        for i, canvas in enumerate(gallery): image.paste(canvas, (0, i*208))
        image.save(args.output.parent / f'{args.dataset}_{pair}.jpg', quality=90)
    write_report(args.output, rows, {'metric': 'cached geometry-v6 mean angle', 'artifact_mask': 'GT-valid pixels only',
        'thresholds': {'near_neutral_vector_length': 0.05, 'GT_smooth_neighbor_deg': 2, 'rough_neighbor_deg': 10},
        'limits': 'Diagnostic thresholds, not benchmark changes. Adjacent-pixel roughness depends on GT resolution. No causal claims. Gallery includes four regressions, two improvements, two median cases.'})


if __name__ == '__main__':
    main()
