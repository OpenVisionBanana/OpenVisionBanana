#!/usr/bin/env python3
"""Training-only distribution audit and reversible single-factor candidates.

Never uses evaluation predictions or changes source archives. Near-duplicate
decisions require agreement in RGB, metric depth and normals within a recorded
trajectory/point.
"""
import argparse
from collections import Counter, defaultdict
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest
from common.surface_normal_encoding import decode_surface_normals_rgb
from common.training.geometry_curation import CONTRACT, source_crop_from_row
from scripts.analysis.audit_zs_geometry_quality import Reader, metric_depth
from scripts.dataset_prep.curate_nzs_geometry import readonly, write_view
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json


def rows(entry):
    with readonly(Path(entry['index_path'])) as db:
        items = [(n, json.loads(p)) for n, p in db.execute('select row_number,payload from rows order by row_number')]
    result = {r['id']:(n, r) for n, r in items}
    if len(result) != len(items): raise ValueError('Duplicate source frame IDs')
    return result


def domain(entry):
    return next(x for x in ('hypersim', 'matterport3d', 'replica', 'synscapes', 'taskonomy') if x in entry['name'])


def group(row):
    m = row.get('metadata', {})
    return (m.get('scene', m.get('building', 'unknown')), m.get('camera', m.get('point_id', 'unknown')))


def shape_stats(valid):
    missing = ~valid
    components, count = label(missing)
    edge_ids = set(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])).tolist()) - {0}
    sizes = np.bincount(components.ravel())[1:]
    return dict(invalid_pixels=int(missing.sum()), invalid_fraction=float(missing.mean()),
        components=int(count), edge_components=len(edge_ids),
        largest_hole_pixels=int(sizes.max()) if len(sizes) else 0,
        interior_components=int(count-len(edge_ids)),
        edge_invalid_fraction=float(np.concatenate((missing[0], missing[-1], missing[:, 0], missing[:, -1])).mean()))


def stable_random(identifiers, count, seed=42):
    return sorted(identifiers, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).digest())[:count]


def confirmed_duplicate(a, b):
    # All signals must agree. Legitimate low-detail views are not rejected on
    # dHash or low Laplacian alone. Metric shapes and native normals count too.
    if min(a['rgb_std'], b['rgb_std']) < 5: return False
    mask = a['mask'] & b['mask']
    if mask.mean() < .8 or np.mean(a['mask'] != b['mask']) > .01: return False
    if np.abs(a['rgb']-b['rgb']).mean() > 2: return False
    relative = np.abs(a['depth']-b['depth']) / np.maximum(a['depth'], .1)
    if np.quantile(relative[mask], .95) > .02: return False
    cosine = (a['normal']*b['normal']).sum(-1)
    return bool(np.quantile(np.degrees(np.arccos(np.clip(cosine[mask], -1, 1))), .95) < 3)


def candidate_manifest(template, sources, selected, destination, name):
    manifest = copy.deepcopy(template)
    replacements = {}
    for dep, nor in sources:
        ids = selected[domain(dep)]
        if not ids: raise ValueError(f'{name}: zero pairs in {domain(dep)}; do not silently remove that dataset')
        for entry in (dep, nor):
            original = rows(entry)
            chosen = [(i, json.dumps(r, sort_keys=True)) for k, (i, r) in original.items() if k in ids]
            if len(chosen) != len(ids): raise ValueError('Candidate IDs missing from one task')
            index = destination/'indices'/f'{entry["name"]}.{name}.sqlite'
            replacements[entry['name']] = write_view(entry, index, chosen, name)
    manifest['datasets'] = [replacements.get(e['name'], e) for e in manifest['datasets']]
    manifest['quality_candidate'] = dict(name=name, seed=42, training_only=True, source_weights_unchanged=True)
    path = destination/'manifests'/f'{name}_full.json'; write_json(path, manifest)
    path.with_suffix('.json.sha256').write_text(f'{digest(path)}  {path.name}\n')
    geometry = copy.deepcopy(manifest); geometry['datasets'] = [e for e in manifest['datasets'] if e['name'] in replacements]
    total = sum(e['weight'] for e in geometry['datasets'])
    for entry in geometry['datasets']: entry['weight'] /= total
    write_json(destination/'manifests'/f'{name}_geometry.json', geometry)
    return dict(manifest=str(path), sha256=digest(path), pairs={k:len(v) for k,v in selected.items()},
                status='CANDIDATE_REQUIRES_PRODUCTION_LOADER_ACCEPTANCE')


def audit(control, paired, full_audit, output, workers=8):
    from train_dreambooth_partial_flux2_klein_img2img import prepared_row_normal_encoding
    if output.exists(): raise FileExistsError('Use a new candidate output version')
    if not (control/'READY.json').is_file(): raise ValueError('Control pool is not accepted')
    template = json.loads((control/'manifests/zeroshot_masked_full.json').read_text())
    entries = [e for e in template['datasets'] if e['name'].startswith(('depth_', 'surface_normals_'))]
    pairs = []
    for d in sorted({domain(e) for e in entries}):
        dep = next(e for e in entries if domain(e)==d and e['name'].startswith('depth_'))
        nor = next(e for e in entries if domain(e)==d and e['name'].startswith('surface_normals_'))
        if rows(dep).keys() != rows(nor).keys(): raise ValueError(f'{d}: task pairs differ')
        pairs.append((dep, nor))
    output.mkdir(); (output/'reports').mkdir(); (output/'indices').mkdir(); (output/'manifests').mkdir()
    provenance = dict(control_manifest=str(control/'manifests/zeroshot_masked_full.json'),
        control_manifest_sha256=digest(control/'manifests/zeroshot_masked_full.json'),
        sources=[dict(name=e['name'], archive=e['path'], index=e['index_path'], index_sha256=digest(Path(e['index_path']))) for e in entries],
        originals_unchanged=True, benchmarks_unchanged=True, seed=42, code_sha256=digest(Path(__file__)))
    write_json(output/'CONTROL.json', provenance)
    zero, diverse, random, decisions, distributions = {}, {}, {}, {}, {}
    for dep, nor in pairs:
        d = domain(dep); dr, nr = rows(dep), rows(nor)
        rd, rn = Reader({**dep, 'offset_base':0}), Reader({**nor, 'offset_base':0})
        zero[d] = set(); diverse[d] = set(dr); random[d] = set(dr)
        summaries = {}; signatures = defaultdict(list)
        def inspect(key):
            row, normalrow = dr[key][1], nr[key][1]
            box = source_crop_from_row(row)
            if box != source_crop_from_row(normalrow): raise ValueError('Paired crop differs')
            dm = rd.image(row, row.get('loss_mask') or row['valid_mask'], 'L')
            nm = rn.image(normalrow, normalrow.get('loss_mask') or normalrow['valid_mask'], 'L')
            valid = (np.asarray(dm) > 0) & (np.asarray(nm) > 0)
            if box:
                l,t,r,b=box; valid=valid[t:b,l:r]
            stats = shape_stats(valid)
            signature = None
            if d in ('hypersim', 'taskonomy'):
                rgb = rd.image(row, row['conditioning_image'])
                depth, method = metric_depth(rd, row, np.asarray(dm)>0)
                if method != 'raw_metric': raise ValueError('Near-duplicate check requires metric raw depth')
                encoding = prepared_row_normal_encoding(normalrow)
                if encoding is None: raise ValueError('No declared normal encoding')
                normal, _ = decode_surface_normals_rgb(np.asarray(rn.image(normalrow, normalrow['image'])), encoding=encoding)
                if box:
                    rgb=rgb.crop(box); depth=depth[t:b,l:r]; normal=normal[t:b,l:r]
                def resize(a): return np.asarray(Image.fromarray(a.astype(np.float32)).resize((32,32), Image.Resampling.NEAREST))
                mask = resize(valid)>0
                depth = resize(np.where(valid, depth, 0))
                normal = np.stack([resize(normal[...,i]) for i in range(3)], -1)
                signature = dict(id=key, group=group(row), rgb=np.asarray(rgb.resize((32,32)), dtype=float),
                    mask=mask, depth=depth, normal=normal, rgb_std=float(np.asarray(rgb.resize((32,32))).mean(-1).std()))
            return key, stats, signature
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for count, (key, stats, signature) in enumerate(ex.map(inspect, sorted(dr)), 1):
                summaries[key] = stats
                if stats['invalid_pixels']==0: zero[d].add(key)
                if signature: signatures[signature['group']].append(signature)
                if count%2048==0: print(f'SAFETY {d}: {count}/{len(dr)}', flush=True)
        duplicate_of = {}
        for trajectory, items in sorted(signatures.items()):
            kept = []
            # At most 20% removal per group preserves view diversity.
            budget = int(len(items)*.2)
            for item in sorted(items, key=lambda r:r['id']):
                matches = [x for x in kept[-8:] if confirmed_duplicate(item, x)]
                if matches and budget:
                    duplicate_of[item['id']] = matches[0]['id']; budget -= 1
                else: kept.append(item)
            removed = set(duplicate_of)&{x['id'] for x in items}
            diverse[d] -= removed
            # Same number per trajectory/point isolates selection from size.
            random[d] -= set(stable_random([x['id'] for x in items], len(removed)))
        decisions[d] = dict(original=len(dr), zero_undefined=len(zero[d]), diverse=len(diverse[d]),
            removed_duplicate_of=duplicate_of, hole_shape_summaries=summaries,
            distribution_fields_not_available=['semantic object categories'] if d=='synscapes' else [])
        write_json(output/'reports'/f'{d}_decisions.json', decisions[d])
        def bins(data):
            result = Counter()
            for key, (_, row) in data.items():
                m=row.get('metadata', {}); scene=m.get('scene', 'unknown')
                if d=='synscapes': scene='synthetic_independent_frames_not_scene_clusters'
                result[f'scene:{scene}']+=1
                for field in ('elevation_index','yaw_index','camera','view_id'):
                    if field in m: result[f'{field}:{m[field]}']+=1
            return dict(result)
        broad=rows(dep)
        strict_manifest=json.loads((control/'manifests/zeroshot_unmasked_full.json').read_text())
        strict_entry=next(e for e in strict_manifest['datasets'] if e['name']==dep['name'])
        strict=rows(strict_entry)
        broad_bins, strict_bins = bins(broad), bins(strict)
        distributions[d] = dict(broad_count=len(broad), strict_count=len(strict),
            retained_fraction=len(strict)/len(broad), broad=broad_bins, strict=strict_bins,
            disappeared_bins=[k for k in broad_bins if not strict_bins.get(k)],
            expected_repeat_multiplier_at_fixed_source_weight=len(broad)/max(1,len(strict)),
            caution='Counts compare curated broad/strict pools, not semantic quality. Unknown fields are not inferred.')
        # Full-scan signals supplement depth-range and invalid-coverage bias.
        scan=full_audit/'scans'/f'{dep["name"]}.sqlite'
        if scan.exists() and d!='replica':
            metrics={'broad':defaultdict(list),'strict':defaultdict(list),'original':defaultdict(list)}
            with readonly(scan) as db:
                for result,payload in db.execute('select result,payload from records'):
                    result=json.loads(result); key=json.loads(payload)['id']; s=result.get('signals',{})
                    for subset in ('original', 'broad', 'strict'):
                        if subset=='broad' and key not in broad: continue
                        if subset=='strict' and key not in strict: continue
                        for field in ('depth_below_0_5m_fraction','rgb_std','laplacian_variance'):
                            if s.get(field) is not None: metrics[subset][field].append(s[field])
                        if s.get('depth_quantiles'): metrics[subset]['median_depth_m'].append(s['depth_quantiles'][2])
            distributions[d]['signal_quantiles']={subset:{k:np.quantile(v,[0,.1,.5,.9,1]).tolist() for k,v in fields.items() if v} for subset,fields in metrics.items()}
        print(f'DONE {d}: zero={len(zero[d])}; diversity={len(diverse[d])}/{len(dr)}',flush=True)
        del signatures
    write_json(output/'reports/distribution_bias.json', distributions)
    registry={}
    for name, selection in [('trajectory_diverse',diverse), ('trajectory_size_control',random)]:
        registry[name]=candidate_manifest(template,pairs,selection,output,name)
    if all(zero.values()): registry['zero_undefined']=candidate_manifest(template,pairs,zero,output,'zero_undefined')
    else: registry['zero_undefined']=dict(status='GATED_EMPTY_DOMAIN',empty_domains=[d for d,v in zero.items() if not v],
        pairs={d:len(v) for d,v in zero.items()},note='Do not silently exclude domains or fill holes to meet this condition')
    write_json(output/'CANDIDATES.json',registry)
    write_json(output/'AUDIT_COMPLETE.json',dict(status='PASS',candidates=registry,
        no_training_started=True,limits='Source-zero validity is not final-loader certification; mandatory preprocessing/loader QA follows.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','paired','full-audit','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8); a=p.parse_args()
    audit(a.control,a.paired,a.full_audit,a.output,a.workers)


if __name__=='__main__': main()
