#!/usr/bin/env python3
"""Repackage one prepared task member into source-specific indexed TARs."""
from __future__ import annotations
import argparse, json, os, shutil, sqlite3, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "scripts" / "dataset_prep" / "build_prepared_tar_index.py"
FIELDS = ("conditioning_image", "image", "loss_mask")

def name_for(source, rows, coconut_xlarge_name=None):
    if source == "coconut_xlarge":
        if coconut_xlarge_name:
            if coconut_xlarge_name == "coconutxl_coco":
                if any("__objects365_v2_" in str(r.get("id", "")) for r in rows):
                    raise ValueError("COCONut-XL COCO member contains Objects365 rows")
            return coconut_xlarge_name
        if not all("__objects365_v2_" in str(r.get("id", "")) for r in rows):
            raise ValueError("COCONut-XL rows are not exclusively Objects365")
        return "coconutxl_objects365"
    return source.removesuffix("_inst")

def paths_for(rows):
    paths = set()
    for row in rows:
        for field in FIELDS:
            if row.get(field):
                path = Path(str(row[field]))
                if path.is_absolute() or ".." in path.parts: raise ValueError(f"Unsafe path: {path}")
                paths.add(path)
    return paths

def previous(archive, index):
    if not archive.exists() and not index.exists(): return None
    if not archive.is_file() or not index.is_file(): raise RuntimeError(f"Incomplete publication: {archive}")
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as con:
        return int(dict(con.execute("SELECT key,value FROM metadata"))["num_rows"])

def publish(root, name, rows):
    archive = root / f"{name}.tar"; index = archive.with_name(archive.name + ".ovb-index.sqlite")
    old = previous(archive, index)
    if old is not None:
        if old != len(rows): raise RuntimeError(f"Existing {name}: {old} rows; expected {len(rows)}")
        return {"name":name,"rows":len(rows),"archive":str(archive),"index_path":str(index),"reused":True}
    member = root / name
    if member.exists(): raise FileExistsError(f"Partial member exists: {member}")
    try:
        member.mkdir()
        for rel in sorted(paths_for(rows)):
            source = root / rel
            if not source.is_file(): raise FileNotFoundError(source)
            target = member / rel; target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
        with (member / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{archive.name}.") as temp:
            temp = Path(temp); a = temp / archive.name; i = temp / index.name
            subprocess.run(["tar","-cf",str(a),"-C",str(root),name],check=True)
            subprocess.run([sys.executable,str(INDEX),str(a),"--output",str(i)],check=True)
            a.replace(archive); i.replace(index)
            sha = a.with_name(a.name + ".sha256")
            if sha.exists(): sha.replace(archive.with_name(archive.name + ".sha256"))
    finally:
        if member.exists(): shutil.rmtree(member)
    return {"name":name,"rows":len(rows),"archive":str(archive),"index_path":str(index),"reused":False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--coconut-xlarge-name",
        choices=("coconutxl_coco", "coconutxl_objects365"),
        help="Published name for the selected COCONut-XL provenance group. "
             "Omit this only for an Objects365-only member (the historic default).",
    )
    args=p.parse_args()
    root=args.member.expanduser().resolve(); groups=defaultdict(list)
    with (root/"metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line); source=str(row.get("source_dataset") or "")
                if not source: raise ValueError("Row has no source_dataset")
                groups[source].append(row)
    members=[]
    for source,rows in sorted(groups.items()):
        if source=="coconut_xlarge" and not args.coconut_xlarge_name:
            # A fresh full pool carries both provenance groups in one member;
            # publish them as the two canonical COCONut-XL tars.
            o365=[r for r in rows if "__objects365_v2_" in str(r.get("id",""))]
            coco=[r for r in rows if "__objects365_v2_" not in str(r.get("id",""))]
            if o365 and coco:
                members.append(publish(root,"coconutxl_objects365",o365))
                members.append(publish(root,"coconutxl_coco",coco))
                continue
        members.append(publish(root,name_for(source, rows, args.coconut_xlarge_name), rows))
    args.report.write_text(json.dumps({"member":str(root),"members":members},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"member":str(root),"members":len(members),"rows":sum(x["rows"] for x in members)}))
if __name__ == "__main__": main()
