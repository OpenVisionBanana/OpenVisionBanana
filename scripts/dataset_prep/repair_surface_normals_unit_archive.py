#!/usr/bin/env python3
"""Version a prepared normal archive while repairing one proven non-unit scene."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.indexed_tar import IndexedTarReference, load_index, sha256_file


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--scene",required=True)
    args=parser.parse_args()
    source=args.source.resolve(); output=args.output.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    rows,_=load_index(source.with_name(source.name+".ovb-index.sqlite"),source)
    selected=[row for row in rows if str((row.get("metadata") or {}).get("scene"))==args.scene]
    target_members={row["image"].member for row in selected if isinstance(row.get("image"),IndexedTarReference)}
    if not target_members: raise ValueError(f"No target rows found for scene {args.scene!r}")
    output.parent.mkdir(parents=True,exist_ok=True); partial=output.with_name(output.name+".partial")
    partial.unlink(missing_ok=True); changed=0
    try:
        with tarfile.open(source,"r:") as src,tarfile.open(partial,"w:") as dst:
            for info in src:
                extracted=src.extractfile(info) if info.isfile() else None
                if info.isfile() and info.name in target_members:
                    image=Image.open(extracted).convert("RGB")
                    rgb=np.asarray(image,dtype=np.float32)
                    normals=rgb/127.5-1.0
                    lengths=np.linalg.norm(normals,axis=-1,keepdims=True)
                    valid=lengths[...,0]>.25
                    repaired=normals.copy(); repaired[valid]/=lengths[valid]
                    encoded=np.clip(np.rint((repaired*.5+.5)*255),0,255).astype(np.uint8)
                    buffer=io.BytesIO(); Image.fromarray(encoded,"RGB").save(buffer,format="PNG")
                    data=buffer.getvalue(); info.size=len(data); extracted=io.BytesIO(data); changed+=1
                elif info.isfile() and Path(info.name).name=="metadata.jsonl":
                    lines=[]
                    for raw in extracted:
                        row=json.loads(raw)
                        if str((row.get("metadata") or {}).get("scene"))==args.scene:
                            row.setdefault("metadata",{})["unit_normal_repair"]="normalize_xyz_before_rgb_encode_v1"
                        lines.append(json.dumps(row,separators=(",",":"),sort_keys=True).encode()+b"\n")
                    data=b"".join(lines); info.size=len(data); extracted=io.BytesIO(data)
                dst.addfile(info,extracted)
        if changed!=len(target_members): raise RuntimeError(f"Expected {len(target_members)} repaired targets, wrote {changed}")
        os.replace(partial,output); output.chmod(0o664)
        digest=sha256_file(output); sidecar=output.with_name(output.name+".sha256")
        sidecar.write_text(f"{digest}  {output.name}\n",encoding="utf-8"); sidecar.chmod(0o664)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps({"source":str(source),"output":str(output),"scene":args.scene,"repaired_rows":changed,"sha256":digest},indent=2))


if __name__=="__main__": main()
