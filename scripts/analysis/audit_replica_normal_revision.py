#!/usr/bin/env python3
"""Audit all published Replica normals against the corrected local revision.

Diagnostic only; original archives, indices and manifests are not changed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.analysis.audit_zs_geometry_quality import Reader, checked_plan, camera_intrinsics
from common.surface_normal_encoding import decode_surface_normals_rgb
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json
from train_dreambooth_partial_flux2_klein_img2img import prepared_row_normal_encoding


def compare_fields(old, new, valid, intrinsics):
    angles = np.degrees(np.arccos(np.clip((old*new).sum(-1), -1, 1)))
    stats = dict(mean_angle=float(angles[valid].mean()),
                 reversed_pixel_fraction=float((angles[valid] > 90).mean()),
                 angle_over5_fraction=float((angles[valid] > 5).mean()))
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        yy, xx = np.mgrid[:valid.shape[0], :valid.shape[1]]
        # Both decoded fields are canonical x-right/y-up/z-toward-viewer.
        ray = np.stack(((xx-cx)/fx, -(yy-cy)/fy, -np.ones_like(xx)), axis=-1)
        for name, field in [('published', old), ('current_v4', new)]:
            dots = (field*ray).sum(-1)
            stats[name+'_away_from_camera_fraction'] = float((dots[valid] > .01).mean())
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); root = args.campaign
    plan = checked_plan(root)
    entries = {e['lane']: e for e in plan['sources']
               if e['dataset'] == 'replica' and (e.get('task') == 'normals' or e['name'].startswith('surface_normals_'))}
    readers = {lane: Reader(e) for lane, e in entries.items()}
    rows = {}
    for lane, e in entries.items():
        with sqlite3.connect('file:'+e['index_path']+'?mode=ro', uri=True) as db:
            rows[lane] = {r['id']: r for payload, in db.execute('select payload from rows') if (r := json.loads(payload))}
    if rows['current'].keys() != rows['published'].keys():
        raise ValueError('Published/current normal frame IDs differ')
    def inspect(sample_id):
        fields = {}; masks = {}
        for lane in ('published', 'current'):
            reader = readers[lane]; row = rows[lane][sample_id]
            rgb = np.asarray(reader.image(row, row['image']))
            fields[lane], vector_valid = decode_surface_normals_rgb(rgb, encoding=prepared_row_normal_encoding(row))
            mask_name = row.get('loss_mask') or row.get('valid_mask') or row.get('mask')
            masks[lane] = (np.asarray(reader.image(row, mask_name, 'L')) > 0) & vector_valid
        valid = masks['published'] & masks['current']
        if not valid.any(): raise ValueError(f'No shared valid normals: {sample_id}')
        h, w = valid.shape
        stats = compare_fields(fields['published'], fields['current'], valid,
                               camera_intrinsics(rows['current'][sample_id], w, h))
        return dict(id=sample_id, mask_difference_fraction=float((masks['published'] != masks['current']).mean()), **stats)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(inspect, sorted(rows['current'])))
    out = root/'reports/normal_revision'; out.mkdir(exist_ok=True)
    with (out/'per_frame.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(records)
    summary = dict(status='FULL_SCAN', frames=len(records),
        whole_or_mostly_reversed_frames=sum(r['reversed_pixel_fraction'] > .5 for r in records),
        frames_with_over1percent_reversed_pixels=sum(r['reversed_pixel_fraction'] > .01 for r in records),
        mean_frame_angle=float(np.mean([r['mean_angle'] for r in records])),
        published_away_fraction_mean=float(np.mean([r['published_away_from_camera_fraction'] for r in records])),
        current_v4_away_fraction_mean=float(np.mean([r['current_v4_away_from_camera_fraction'] for r in records])),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caution='Comparison against the corrected depth-derived revision, not independent mesh-normal ground truth',
        originals_unchanged=True)
    write_json(out/'summary.json', summary)
    selected = sorted(records, key=lambda r: r['reversed_pixel_fraction'], reverse=True)[:12]
    canvas = Image.new('RGB', (768, len(selected)*215), 'white'); draw = ImageDraw.Draw(canvas)
    for i, stats in enumerate(selected):
        sample_id = stats['id']; row = rows['published'][sample_id]
        images = [readers['published'].image(row, row['conditioning_image']),
                  readers['published'].image(row, row['image']),
                  readers['current'].image(rows['current'][sample_id], rows['current'][sample_id]['image'])]
        for j, image in enumerate(images):
            image.thumbnail((250, 180)); canvas.paste(image, (j*256, i*215))
        draw.text((3, i*215+185), f'{sample_id}: mean={stats["mean_angle"]:.1f} deg, reversed={stats["reversed_pixel_fraction"]:.1%}', fill='black')
        draw.text((3, i*215+200), 'RGB | published normal target | corrected local normal target', fill='black')
    canvas.save(out/'reversed_targets.jpg', quality=92)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
