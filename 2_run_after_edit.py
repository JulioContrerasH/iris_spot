#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy IRIS mask previews to a single folder, renamed as <ID>.png.

For each ID in IDS, this script looks for:
  <SPOT_ROOT>/<ID>/<ID>.iris/segmentation/<ID>/mask.png
and copies it to:
  <OUTPUT_DIR>/<ID>.png

Configure SPOT_ROOT, OUTPUT_DIR, and IDS below.
"""

from pathlib import Path
import shutil


# ------------- CONFIG (edit these) -------------
SPOT_ROOT  = Path("spot")                      # base folder where all <ID> folders-projects live
OUTPUT_DIR = Path("save_mask")      # where renamed PNGs will be saved
IDS = [
    "V1KRNP____19990110F185_V003",
    # "ADD_MORE_IDS_HERE",
]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ok, missing = 0, 0
for id_name in IDS:
    # Expected source: spot/<ID>/<ID>.iris/segmentation/<ID>/mask.png
    src = SPOT_ROOT / id_name / f"{id_name}.iris" / "segmentation" / id_name / "mask.png"
    dst = OUTPUT_DIR / f"{id_name}.png"

    if not src.exists():
        print(f"[MISS] {id_name}: not found -> {src}")
        missing += 1
        continue

    # Copy (overwrite if exists) and rename to <ID>.png
    shutil.copyfile(src, dst)
    print(f"[OK]   {id_name}: -> {dst}")