#!/usr/bin/env python3
"""Report fixed-pose colour diagnostics without selecting a training winner."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from common.training.asset_lock import digest
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    root=p.parse_args().campaign/'ptex_colour_candidates';results={};ids=None
    for name in ('official','half_exposure','half_exposure_neutral_saturation'):
        path=root/name/'acceptance.json'
        if not path.exists():
            print(f'GATED: incomplete fixed-pose colour probe {name}');return
        data=json.loads(path.read_text());frames=data['frames']
        frame_ids=[f['id'] for f in frames]
        if ids is not None and frame_ids!=ids:raise ValueError('Colour comparison camera IDs differ')
        ids=frame_ids
        results[name]=dict(report=str(path),report_sha256=digest(path),registration_passed=data['registration_passed'],
            frames=len(frames),fixed_scales=data['fixed_colour_scales'],
            mean_channel_highlight_fraction=mean(f['ptex_extreme_fraction'] for f in frames),
            mean_channel_dark_fraction=mean(f['ptex_dark_fraction'] for f in frames),
            sdk_settings=sorted({f['sdk_colour_settings'] for f in frames}))
    write_json(root/'COMPARISON.json',dict(status='REGISTERED_REQUIRES_COLOUR_PILOT' if all(r['registration_passed'] for r in results.values()) else 'GATED_REGISTRATION',
        variants=results,training_winner=None,production_adopted=False,
        limitations='Brightness/clipping proxies do not certify realism or training gains. Fixed settings only; no benchmark-driven colour selection, per-image equalization, or GT modification.'))


if __name__=='__main__':main()
