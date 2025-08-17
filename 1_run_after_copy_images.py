#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-entry script that:
- Builds IRIS projects per ID using base.json as template.
- In each project, puts ONLY two files under spot/<ID>/images/<ID>:
    * rgb.npy   (created from source rgb.tif)
    * metadata.json (center lat/lon)
- No metadata.json is written under spot/images/<ID>.
- No rgb.npy is kept in source; if found at spot/images/<ID>/rgb.npy, it is removed.

Result (per ID):
  spot/<ID>/
    <ID>.json
    images/<ID>/
      rgb.npy
      metadata.json
    <ID>.iris/segmentation/<ID>/{1_final.npy, 1_user.npy, mask.png}  # from f_1dpwseg.tif if exists

Run:
  python run_iris_prep.py
Then:
  iris label spot/<ID>/<ID>.json
"""

from pathlib import Path
import os, json
import numpy as np
import rasterio as rio
from rasterio.warp import transform
from PIL import Image

# ---------------- CONFIG ----------------
BASE_JSON       = "base.json"          # base template for per-ID project
IMAGES_ROOT     = Path("spot/images")  # where {id}/rgb.tif (and f_1dpwseg.tif) live
MASK_TIF_NAME   = "f_1dpwseg.tif"      # input mask values may be 0/1/255 (255 will be remapped to 0)
OVERWRITE_JSON  = True                 # overwrite <ID>.json if exists
# Two-class setup: 0 -> Clear, 1 -> Cloud. No "NoData" class anymore.
CLASS_MAP = {0: "Clear", 1: "Cloud"}
# ---------------------------------------


# --------- Small helpers ---------
def list_ids(root: Path):
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def write_rgb_npy_to_dest(tif_path: Path, dest_npy: Path):
    """Create rgb.npy directly in destination from rgb.tif (H,W,C float32)."""
    if dest_npy.is_symlink() or dest_npy.exists():
        dest_npy.unlink()
    with rio.open(tif_path) as src:
        arr = src.read()                            # (C,H,W)
    arr = np.moveaxis(arr, 0, -1).astype("float32") # (H,W,C)
    dest_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest_npy, arr, allow_pickle=False)
    print(f"  [+] {dest_npy.parent.parent.name}: dest rgb.npy created")


def get_width_height_from_tif(tif_path: Path):
    with rio.open(tif_path) as src:
        return src.width, src.height  # (W,H)


def compute_center_latlon(tif_path: Path):
    """Return (lat, lon) of raster center in WGS84. None if CRS missing."""
    with rio.open(tif_path) as src:
        if src.crs is None:
            return None
        cx = (src.bounds.left + src.bounds.right) / 2
        cy = (src.bounds.bottom + src.bounds.top) / 2
        lon, lat = transform(src.crs, "EPSG:4326", [cx], [cy])
        return float(lat[0]), float(lon[0])


def safe_make_clean_dest_dir(path: Path):
    """
    Ensure path is a real directory (not a symlink to source).
    If a symlink exists from old runs, remove it and create a real dir.
    """
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def remove_source_rgb_npy(id_src_dir: Path):
    """Ensure no rgb.npy remains in source tree."""
    src_npy = id_src_dir / "rgb.npy"
    try:
        if src_npy.is_symlink() or src_npy.exists():
            src_npy.unlink()
            print(f"  [-] {id_src_dir.name}: removed source rgb.npy")
    except Exception as e:
        print(f"  [WARN] {id_src_dir.name}: could not remove source rgb.npy -> {e}")


# --------- IRIS caches from mask (2-class remap) ---------
def build_caches_from_mask(id_src_dir: Path, proj_dir: Path, proj_name: str,
                           classes: list[str], class_colors: list[tuple[int,int,int]]):
    """
    Read f_1dpwseg.tif from SOURCE and create in PROJECT:
      <proj>/<name>.iris/segmentation/<ID>/1_final.npy  (H,W,K) bool
      <proj>/<name>.iris/segmentation/<ID>/1_user.npy   (H,W)   bool
      mask.png (colored visualization)

    Two-class policy:
      - Remap 255 -> 0 (i.e., treat 255 as Clear).
      - Map value 0 to Clear; value 1 to Cloud.
      - Any other value (2..254) is treated as Cloud as a fallback.
      - user mask is fully valid (all True), since there is no NoData.
    """
    tif_mask = id_src_dir / MASK_TIF_NAME
    if not tif_mask.exists():
        return

    with rio.open(tif_mask) as src:
        m = src.read(1).astype("uint8")

    # --- Remap 255 -> 0 to drop NoData and keep only two classes ---
    m = np.where(m == 255, 0, m)

    H, W = m.shape
    K = len(classes)
    onehot = np.zeros((H, W, K), dtype=bool)

    # Map 0 -> Clear
    if "Clear" in classes:
        k_clear = classes.index("Clear")
        onehot[:, :, k_clear] |= (m == 0)

    # Map 1 -> Cloud and also treat any 2..254 as Cloud (fallback)
    if "Cloud" in classes:
        k_cloud = classes.index("Cloud")
        onehot[:, :, k_cloud] |= (m == 1)
        onehot[:, :, k_cloud] |= (m > 1)  # values 2..254 become Cloud

    # All pixels are valid now (no NoData)
    user = np.ones((H, W), dtype=bool)

    seg_dir = proj_dir / f"{proj_name}.iris" / "segmentation" / id_src_dir.name
    seg_dir.mkdir(parents=True, exist_ok=True)
    np.save(seg_dir / "1_final.npy", onehot)
    np.save(seg_dir / "1_user.npy",  user)

    # Color preview
    rgbmask = np.zeros((H, W, 3), dtype=np.uint8)
    for k, _n in enumerate(classes):
        rgbmask[onehot[:, :, k]] = class_colors[k]
    Image.fromarray(rgbmask, "RGB").save(seg_dir / "mask.png", optimize=True)


# --------- Project creation ---------
def create_projects_per_id(base_json_path: Path, images_root: Path):
    if not base_json_path.exists():
        raise FileNotFoundError(f"Missing {base_json_path.name}")
    if not images_root.exists():
        raise FileNotFoundError(f"Missing {images_root}")

    with open(base_json_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)
    classes = [c["name"] for c in base_cfg["classes"]]
    class_colors = [tuple(c["colour"][:3]) for c in base_cfg["classes"]]

    ids = list_ids(images_root)
    if not ids:
        print("No IDs in spot/images. Nothing to do.")
        return

    run_lines = []

    for id_name in ids:
        id_src_dir = images_root / id_name
        tif_rgb = id_src_dir / "rgb.tif"
        if not tif_rgb.exists():
            print(f"[{id_name}] no rgb.tif -> skip")
            continue

        # shape from source TIFF
        W, H = get_width_height_from_tif(tif_rgb)
        print(f"[{id_name}] shape: {W}x{H}")

        # project folder
        proj_dir = base_json_path.parent / "spot" / id_name
        proj_dir.mkdir(parents=True, exist_ok=True)

        # project images/<ID>: real directory (no symlinked tree)
        proj_images_id_dir = proj_dir / "images" / id_name
        safe_make_clean_dest_dir(proj_images_id_dir)

        # create rgb.npy ONLY in DEST
        dest_npy = proj_images_id_dir / "rgb.npy"
        write_rgb_npy_to_dest(tif_rgb, dest_npy)

        # write metadata.json ONLY in DEST
        latlon = compute_center_latlon(tif_rgb)
        if latlon is None:
            print(f"[{id_name}] WARNING: no CRS -> create metadata.json manually with 'location: [lat,lon]'")
        else:
            lat, lon = latlon
            meta = {"scene_id": id_name, "location": [lat, lon]}
            (proj_images_id_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[{id_name}] metadata.json -> location {lon:.6f}, {lat:.6f}")

        # ensure no rgb.npy remains in SOURCE
        remove_source_rgb_npy(id_src_dir)

        # per-ID JSON (deep copy of base)
        proj_json = proj_dir / f"{id_name}.json"
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        cfg["name"] = id_name
        cfg["images"]["path"] = "images/{id}/rgb.npy"
        cfg["images"]["shape"] = [W, H]
        cfg.setdefault("segmentation", {})
        cfg["segmentation"]["mask_area"] = [0, 0, W, H]

        if OVERWRITE_JSON or not proj_json.exists():
            with open(proj_json, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"  [+] JSON: {proj_json}")

        # caches from mask (SOURCE -> PROJECT, 2-class)
        build_caches_from_mask(id_src_dir, proj_dir, id_name, classes, class_colors)

        run_lines.append(f"iris label {proj_json}")

    # launcher script
    run_sh = base_json_path.parent / "run_all_projects.sh"
    with open(run_sh, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -e\n\n")
        for line in run_lines:
            f.write(line + "\n")
    os.chmod(run_sh, 0o755)

    print("\nDone.")
    print("Launch a single project with:")
    if run_lines:
        print(" ", run_lines[0])
    print(f"Or run all with: {run_sh}")


# --------- Main entry ---------
def main():
    # Build per-ID IRIS projects; rgb.npy exists only in destination.
    create_projects_per_id(Path(BASE_JSON).resolve(), IMAGES_ROOT.resolve())


if __name__ == "__main__":
    main()
