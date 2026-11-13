#!/usr/bin/env python3
"""Export faithful old/new Replica RGB previews and paired GT for visual review."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.recheck_replica_publication import load
from common.indexed_tar import read_reference_bytes

REPRESENTATIVE = (
    'replica_apartment_0_000254_upright',
    'replica_office_1_000245_upright',
    'replica_room_0_000265_upright',
    'replica_apartment_2_000253_upright',
    'replica_hotel_0_000252_upright',
    'replica_office_3_000248_upright',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/replica_ptex_fix_20261004'))
    p.add_argument('--old', type=Path, default=Path('data/geometry_training_publication_20261004/upload'))
    p.add_argument('--source', type=Path, default=Path('data/replica_upright_quality_20261002/data/archives'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new preview directory; no datasets are modified')
    if json.loads((a.root/'upload/READY.json').read_text())['status'] != 'materialized_export_complete':
        raise ValueError('Corrected package is not ready')
    a.output.mkdir(parents=True)
    (a.output/'native_rgb').mkdir()
    rows, readers = [], []
    for base, name in ((a.old, 'depth_legacy_replica-00000'),
                       (a.root/'upload', 'depth_legacy_replica-00000'),
                       (a.root/'upload', 'surface_normals_legacy_replica_camera_facing_v4-00000')):
        r, reader = load(base/'archives'/(name+'.tar'), base/'indices'/(name+'.tar.ovb-index.sqlite'))
        rows.append(r); readers.append(reader)
    sr, sd = load(a.source/'replica_depth_upright.tar', a.source/'replica_depth_upright.tar.ovb-index.sqlite')
    assert set(rows[0]) == set(rows[1]) == set(rows[2]) == set(sr)
    frames = [json.loads(line) for line in (a.root/'qa/frames.jsonl').read_text().splitlines()]
    variant = 'half_exposure_neutral_saturation'
    difficult = list(dict.fromkeys(max(frames, key=lambda f: f['colours'][variant][key])['id']
                     for key in ('fully_white_pixels', 'fully_black_pixels', 'clipped_channels')))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        smallfont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = smallfont = ImageFont.load_default()
    records = []
    for sample in dict.fromkeys((*REPRESENTATIVE, *difficult)):
        for label, r, reader in zip(('old', 'corrected'), rows, readers):
            row = r[sample]
            blob = read_reference_bytes(reader.ref(row, row['conditioning_image']))
            # Export the original PNG bytes. No exposure adjustment, filtering,
            # sharpening, interpolation or conversion is applied to these files.
            (a.output/'native_rgb'/f'{sample}_{label}.png').write_bytes(blob)
            records.append(dict(id=sample, variant=label, png_sha256=hashlib.sha256(blob).hexdigest()))

    def rgb_page(ids, title, filename):
        cell, rowheight = 512, 550
        canvas = Image.new('RGB', (1024, 80+len(ids)*rowheight), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), title, fill='black', font=font)
        draw.text((12, 40), 'OLD: vertex colours', fill='black', font=font)
        draw.text((524, 40), 'CORRECTED: official Ptex textures', fill='black', font=font)
        for i, sample in enumerate(ids):
            y = 80+i*rowheight
            for k, (r, reader) in enumerate(zip(rows[:2], readers[:2])):
                row = r[sample]
                im = reader.image(row, row['conditioning_image'])
                assert im.size == (cell, cell)
                canvas.paste(im, (k*cell, y))
            draw.text((12, y+515), sample+' | identical camera and GT', fill='black', font=smallfont)
        canvas.save(a.output/filename, quality=97, subsampling=0)

    rgb_page(REPRESENTATIVE[:3], 'Replica RGB: apartment, office and room', 'rgb_before_after_1.jpg')
    rgb_page(REPRESENTATIVE[3:], 'Replica RGB: three more scenes', 'rgb_before_after_2.jpg')
    rgb_page(difficult, 'Review cases: highest clipping / darkest shadows (not filtered out)', 'rgb_review_cases.jpg')
    from matplotlib import colormaps
    canvas = Image.new('RGB', (1024, 70+len(REPRESENTATIVE)*285), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'Corrected Replica training pairs: ground truth, not model predictions', fill='black', font=font)
    for k, title in enumerate(('Corrected input RGB', 'Metric depth (display)', 'Stored normal GT (VB)', 'Joint validity mask')):
        draw.text((k*256+6, 40), title, fill='black', font=smallfont)
    for i, sample in enumerate(REPRESENTATIVE):
        dep, nor = rows[1][sample], rows[2][sample]
        rgb = readers[1].image(dep, dep['conditioning_image'])
        normal = readers[2].image(nor, nor['image'])
        valid = np.asarray(readers[1].image(dep, dep['loss_mask'], 'L')) > 0
        nvalid = np.asarray(readers[2].image(nor, nor['loss_mask'], 'L')) > 0
        depth = sd.array(sr[sample], sr[sample]['depth_raw'])
        lo, hi = float(depth[valid].min()), float(depth[valid].max())
        display = np.clip((np.nan_to_num(depth, nan=lo)-lo)/max(hi-lo, 1e-8), 0, 1)
        colour = (colormaps['Spectral'](display)[..., :3]*255).astype('uint8')
        colour[~valid] = (128, 128, 128)
        mask = Image.fromarray((valid & nvalid).astype('uint8')*255).convert('RGB')
        y = 70+i*285
        for k, im in enumerate((rgb, Image.fromarray(colour), normal, mask)):
            mode = Image.Resampling.LANCZOS if k == 0 else Image.Resampling.NEAREST
            canvas.paste(im.resize((256, 256), mode), (k*256, y))
        draw.text((8, y+260), f'{sample} | depth {lo:.2f}-{hi:.2f} m | joint invalid {100*(~(valid & nvalid)).mean():.3f}%',
                  fill='black', font=smallfont)
    canvas.save(a.output/'paired_geometry.jpg', quality=97, subsampling=0)
    (a.output/'SAMPLES.json').write_text(json.dumps(dict(representative_ids=REPRESENTATIVE,
        review_ids=difficult, selection='Same six diverse-scene examples as earlier review, plus QA extrema; not random or exhaustive',
        native_rgb_bytes_unmodified=True, previews_only=True, png_records=records), indent=2)+'\n')
    print(f'Exported {len(records)//2} old/new RGB pairs and four galleries to {a.output}', flush=True)


if __name__ == '__main__':
    main()
