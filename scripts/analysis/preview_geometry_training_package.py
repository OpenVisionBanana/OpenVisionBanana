#!/usr/bin/env python3
"""Read-only visual samples from the exact materialized training package."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.indexed_tar import IndexedTarReference, read_reference_bytes


def fit(im, size):
    # Preserve aspect ratio. White letterboxing belongs to this preview only.
    im = ImageOps.contain(im, size, Image.Resampling.NEAREST)
    canvas = Image.new('RGB', size, 'white')
    canvas.paste(im, ((size[0]-im.width)//2, (size[1]-im.height)//2))
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a fresh output directory; no dataset is modified')
    if not (a.package/'READY.json').is_file():
        raise ValueError('Package READY marker missing')
    entries = json.loads((a.package/'manifests/base_geometry_indexed.json').read_text())['datasets']
    groups = {}
    for entry in entries:
        index = a.package/entry['index_path']
        with sqlite3.connect('file:'+str(index)+'?mode=ro', uri=True) as c:
            # Read small metadata only; never extract or scan entire TAR payloads.
            for rowid, dataset, task, invalid in c.execute(
                    "select rowid,json_extract(payload,'$.source_dataset'),"
                    "json_extract(payload,'$.task'),"
                    "json_extract(payload,'$.metadata.invalid_fraction') from rows order by rowid"):
                if dataset == 'replica':
                    continue
                groups.setdefault((dataset, task), []).append((entry, rowid, invalid))
    a.output.mkdir(parents=True)
    (a.output/'native').mkdir()
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = small = ImageFont.load_default()
    report = dict(package=str(a.package),
        package_sha256=hashlib.sha256((a.package/'PACKAGE.json').read_bytes()).hexdigest(),
        selection='Four evenly-spaced row positions, plus two highest recorded invalid fractions per dataset/task. Not random, exhaustive or a quality certificate.',
        displays='Unmodified stored RGB/GT/mask bytes exported separately. Galleries preserve aspect ratio, use white letterboxing and nearest-neighbour resizing. No smoothing, recolouring or invalid-pixel overlay.',
        galleries=[])
    for (dataset, task), candidates in sorted(groups.items()):
        chosen = []
        seen = set()
        for pos in np.linspace(0, len(candidates)-1, min(4, len(candidates)), dtype=int):
            entry, rowid, invalid = candidates[int(pos)]
            chosen.append((entry, rowid, 'spread sample'))
            seen.add((entry['name'], rowid))
        for entry, rowid, invalid in sorted(candidates, key=lambda r: float(r[2] or 0), reverse=True):
            if (entry['name'], rowid) not in seen:
                chosen.append((entry, rowid, 'high-invalid review'))
                seen.add((entry['name'], rowid))
            if len(chosen) >= 6:
                break
        normal = task != 'depth_estimation'
        slug = dataset + ('_normals' if normal else '_depth')
        canvas = Image.new('RGB', (1200, 82+len(chosen)*330), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f'{dataset}: stored training pairs — ground truth, not predictions', fill='black', font=font)
        for k, title in enumerate(('Input RGB', 'Normal GT (stored VB RGB)' if normal else 'Depth GT (stored VB RGB)', 'Loss mask: white = valid')):
            draw.text((k*400+10, 45), title, fill='black', font=font)
        records = []
        for i, (entry, rowid, selection) in enumerate(chosen):
            with sqlite3.connect('file:'+str(a.package/entry['index_path'])+'?mode=ro', uri=True) as c:
                row = json.loads(c.execute('select payload from rows where rowid=?', (rowid,)).fetchone()[0])
                blobs, images, hashes = [], [], {}
                for field, suffix in (('conditioning_image', 'rgb'), ('image', 'gt'), ('loss_mask', 'mask')):
                    member = row[field]
                    result = c.execute('select offset,size from members where name=?', (member,)).fetchone()
                    if result is None:
                        raise ValueError(f'Missing member: {member}')
                    offset, size = result
                    blob = read_reference_bytes(IndexedTarReference(str(a.package/entry['path']), member, offset, size))
                    target = a.output/'native'/f'{slug}_{i:02d}_{suffix}.png'
                    target.write_bytes(blob)
                    hashes[suffix] = hashlib.sha256(blob).hexdigest()
                    with Image.open(io.BytesIO(blob)) as im:
                        im.load()
                        images.append(im.copy())
            rgb, gt, mask = images
            if rgb.size != gt.size or gt.size != mask.size:
                raise ValueError(f'Unregistered dimensions for {row["id"]}')
            m = np.asarray(mask.convert('L'))
            if not set(np.unique(m)).issubset({0, 255}):
                raise ValueError(f'Non-binary mask: {row["id"]}')
            invalid = float((m == 0).mean())
            y = 82+i*330
            for k, im in enumerate(images):
                canvas.paste(fit(im.convert('RGB'), (390, 280)), (k*400+5, y))
            draw.text((10, y+285), f'{i+1}. {row["id"]} | {selection}', fill='black', font=small)
            draw.text((10, y+307), f'{rgb.width}x{rgb.height} | invalid {invalid*100:.3f}% | {entry["name"]}', fill='black', font=small)
            records.append(dict(id=row['id'], selection=selection, invalid_fraction=invalid,
                archive=entry['path'], index=entry['index_path'], rowid=rowid,
                native_prefix=f'native/{slug}_{i:02d}', hashes=hashes, metadata=row.get('metadata', {})))
        filename = slug+'.jpg'
        canvas.save(a.output/filename, quality=97, subsampling=0)
        report['galleries'].append(dict(dataset=dataset, task=task, file=filename,
            total_training_rows=len(candidates), samples=records))
        print(f'{slug}: {len(records)} samples / {len(candidates)} rows', flush=True)
    (a.output/'SAMPLES.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Other geometry-training datasets: visual samples', '',
        'These are exact samples from the current corrected ZS + base-clean NZS package, not model predictions.', '',
        'Each gallery has input RGB, the stored RGB training target, and the binary loss mask (white valid, black invalid).',
        'Depth is the stored Vision Banana RGB encoding, not a metric-depth colour scale. Normals retain their stored encoding.',
        'White padding outside rectangular images is preview letterboxing, not dataset invalid regions.', '',
        'Four spread examples and two higher-invalid review examples are selected per dataset/task. ',
        'The review examples are worst by recorded invalid fraction, not necessarily worst visual quality.',
        'Equal invalid fractions can make their selection arbitrary. These samples cannot certify the whole dataset.', '',
        'Original image/GT/mask bytes are in `native/`; exact provenance, SHA-256 hashes and sample IDs are in `SAMPLES.json`.', '',
        '| Dataset | Depth | Normals |', '|---|---|---|']
    for dataset in sorted(set(g['dataset'] for g in report['galleries'])):
        lines.append(f'| {dataset} | [View]({dataset}_depth.jpg) | [View]({dataset}_normals.jpg) |')
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
