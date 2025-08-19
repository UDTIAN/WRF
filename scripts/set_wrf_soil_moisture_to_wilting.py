#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from glob import glob

import numpy as np
from netCDF4 import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set soil moisture to the wilting point for land grid points within a lat/lon box "
            "in WRF NetCDF files (wrfinput_*, wrflowinp_*, wrfout_*)."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help=(
            "File paths or glob patterns to NetCDF files to modify. "
            "Examples: /data/wrfinput_d01 /data/wrflowinp_d01 '/data/wrfout_d01_*'"
        ),
    )
    parser.add_argument("--lat-min", type=float, default=25.0, help="Minimum latitude (deg N)")
    parser.add_argument("--lat-max", type=float, default=40.0, help="Maximum latitude (deg N)")
    parser.add_argument("--lon-min", type=float, default=-125.0, help="Minimum longitude (deg E, negative for W)")
    parser.add_argument("--lon-max", type=float, default=-100.0, help="Maximum longitude (deg E, negative for W)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not modify files; only print what would change.")
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Do not create .bak backup copies before modifying files.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.")
    parser.set_defaults(backup=True)
    return parser.parse_args()


def expand_files(patterns: list[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            expanded.extend(matches)
        else:
            # Treat as literal path if exists; otherwise keep pattern to error later
            if os.path.exists(pattern):
                expanded.append(pattern)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for path in expanded:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def find_var(nc: Dataset, names: list[str]) -> str | None:
    for name in names:
        if name in nc.variables:
            return name
    return None


def to_0360(lon_deg: np.ndarray) -> np.ndarray:
    return np.mod(lon_deg, 360.0)


def build_region_mask(nc: Dataset, lat_min: float, lat_max: float, lon_min: float, lon_max: float, verbose: bool) -> np.ndarray:
    lat_name = find_var(nc, ["XLAT", "XLAT_M"]) or "XLAT"
    lon_name = find_var(nc, ["XLONG", "XLONG_M"]) or "XLONG"
    if lat_name not in nc.variables or lon_name not in nc.variables:
        raise RuntimeError("Could not find XLAT/XLAT_M and XLONG/XLONG_M in file.")

    lat_var = nc.variables[lat_name]
    lon_var = nc.variables[lon_name]

    # Expect shapes like (Time, south_north, west_east)
    lat = np.array(lat_var)
    lon = np.array(lon_var)
    if lat.ndim == 3:
        lat2d = lat[0]
        lon2d = lon[0]
    elif lat.ndim == 2:
        lat2d = lat
        lon2d = lon
    else:
        raise RuntimeError(f"Unexpected XLAT dims: {lat.shape}")

    # Determine if longitudes are in 0-360 or -180..180
    lon2d_min = float(np.nanmin(lon2d))
    lon2d_max = float(np.nanmax(lon2d))
    use_0360 = lon2d_min >= 0.0 and lon2d_max <= 360.0

    req_lon_min = lon_min
    req_lon_max = lon_max
    if use_0360:
        req_lon_min = (lon_min + 360.0) if lon_min < 0.0 else lon_min
        req_lon_max = (lon_max + 360.0) if lon_max < 0.0 else lon_max

    if verbose:
        sys.stderr.write(
            f"Lon system: {'0-360' if use_0360 else '-180..180'}; "
            f"target box lat[{lat_min},{lat_max}] lon[{req_lon_min},{req_lon_max}]\n"
        )

    lat_mask = (lat2d >= lat_min) & (lat2d <= lat_max)
    lon_mask = (lon2d >= req_lon_min) & (lon2d <= req_lon_max)
    region_mask = lat_mask & lon_mask
    return region_mask


def find_land_mask(nc: Dataset) -> np.ndarray:
    if "LANDMASK" not in nc.variables:
        # Try to infer from LU_INDEX if LANDMASK is missing
        if "LU_INDEX" in nc.variables:
            lu = np.array(nc.variables["LU_INDEX"])
            if lu.ndim == 3:
                lu2d = lu[0]
            elif lu.ndim == 2:
                lu2d = lu
            else:
                raise RuntimeError(f"Unexpected LU_INDEX dims: {lu.shape}")
            # Treat non-zero LU_INDEX as land
            return (lu2d != 0).astype(bool)
        else:
            raise RuntimeError("LANDMASK not found and LU_INDEX unavailable to infer land points.")
    lm = np.array(nc.variables["LANDMASK"])  # 1 for land, 0 for water
    if lm.ndim == 3:
        return lm[0].astype(bool)
    if lm.ndim == 2:
        return lm.astype(bool)
    raise RuntimeError(f"Unexpected LANDMASK dims: {lm.shape}")


def get_wilt_field(nc: Dataset, target_shape_3d: tuple[int, int, int]) -> np.ndarray:
    if "WILT" not in nc.variables:
        raise RuntimeError("WILT not found in file. Provide files with WILT or modify the script to use a constant.")

    wilt = np.array(nc.variables["WILT"])  # Could be (soil_layers, y, x) or (y, x)
    if wilt.ndim == 3:
        return wilt
    if wilt.ndim == 2:
        nsoil = target_shape_3d[0]
        wilt3d = np.repeat(wilt[None, ...], nsoil, axis=0)
        return wilt3d
    raise RuntimeError(f"Unexpected WILT dims: {wilt.shape}")


def update_var_to_wilt(nc: Dataset, var_name: str, region_land_mask: np.ndarray, wilt3d: np.ndarray, verbose: bool) -> tuple[int, int]:
    var = nc.variables[var_name]
    dims = var.dimensions

    # Determine indices for standard dims
    dim_names = list(dims)
    has_time = "Time" in dim_names
    time_len = var.shape[dim_names.index("Time")] if has_time else 1

    if "soil_layers_stag" in dim_names:
        k_idx = dim_names.index("soil_layers_stag")
    else:
        raise RuntimeError(f"{var_name} missing soil_layers_stag dimension: dims={dim_names}")

    if "south_north" in dim_names and "west_east" in dim_names:
        j_idx = dim_names.index("south_north")
        i_idx = dim_names.index("west_east")
    else:
        raise RuntimeError(f"{var_name} missing horizontal dims: dims={dim_names}")

    # Prepare mask shaped like (k, j, i)
    nsoil = var.shape[k_idx]
    ny = var.shape[j_idx]
    nx = var.shape[i_idx]
    if wilt3d.shape != (nsoil, ny, nx):
        raise RuntimeError(
            f"WILT shape {wilt3d.shape} does not match soil/horizontal dims {(nsoil, ny, nx)}"
        )
    mask2d = region_land_mask  # (j, i)
    if mask2d.shape != (ny, nx):
        raise RuntimeError(f"Mask shape {mask2d.shape} does not match horizontal dims {(ny, nx)}")
    mask3d = np.broadcast_to(mask2d, (nsoil, ny, nx))

    # We'll iterate in slices to avoid reordering dimensions; build index slices accordingly
    changed_points = 0
    total_points = int(mask3d.sum()) * (time_len if has_time else 1)

    # Prepare a function to build slicing tuple with wildcards for other dims
    def build_slice(t: int | None, k: int | slice, j_sel: np.ndarray, i_sel: np.ndarray):
        sl: list[object] = [slice(None)] * len(dim_names)
        if has_time and t is not None:
            sl[dim_names.index("Time")] = t
        sl[k_idx] = k
        sl[j_idx] = j_sel
        sl[i_idx] = i_sel
        return tuple(sl)

    # Compute indices where mask is true
    jj, ii = np.where(mask2d)
    if jj.size == 0:
        if verbose:
            sys.stderr.write(f"No grid points in target region for {var_name}.\n")
        return (0, 0)

    # For efficiency, group by rows
    by_row: dict[int, np.ndarray] = {}
    for j, i in zip(jj, ii):
        by_row.setdefault(j, []).append(i)
    # Convert lists to arrays
    for j in list(by_row.keys()):
        by_row[j] = np.asarray(by_row[j], dtype=int)

    # Update values
    for t in range(time_len):
        for k in range(nsoil):
            wilt_row = wilt3d[k]
            for j, cols in by_row.items():
                target_slice = build_slice(t if has_time else None, k, j, cols)
                var_vals = var[target_slice]
                new_vals = wilt_row[j, cols].astype(var.dtype)
                # Count changes
                if verbose:
                    changed_points += int(new_vals.size)
                var[target_slice] = new_vals

    return (changed_points, total_points)


def process_file(path: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float, do_backup: bool, dry_run: bool, verbose: bool) -> None:
    if verbose:
        sys.stderr.write(f"Processing {path}\n")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Backup
    if do_backup and not dry_run:
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)
            if verbose:
                sys.stderr.write(f"  Backed up to {backup_path}\n")

    mode = "r" if dry_run else "r+"
    with Dataset(path, mode) as nc:
        region_mask = build_region_mask(nc, lat_min, lat_max, lon_min, lon_max, verbose)
        land_mask = find_land_mask(nc)
        region_land_mask = region_mask & land_mask

        # Identify target variables
        target_vars = [name for name in ("SMOIS", "SH2O") if name in nc.variables]
        if not target_vars:
            raise RuntimeError("None of SMOIS/SH2O found in file.")

        # Build a reference 3D shape using first target var
        ref = nc.variables[target_vars[0]]
        dims = ref.dimensions
        if "soil_layers_stag" not in dims or ("south_north" not in dims or "west_east" not in dims):
            raise RuntimeError(f"Unexpected dims for {target_vars[0]}: {dims}")
        nsoil = ref.shape[dims.index("soil_layers_stag")]
        ny = ref.shape[dims.index("south_north")]
        nx = ref.shape[dims.index("west_east")]
        wilt3d = get_wilt_field(nc, (nsoil, ny, nx))

        if dry_run:
            selected = int((region_land_mask).sum())
            sys.stderr.write(
                f"  Will set {', '.join(target_vars)} to WILT on {selected} land points (per soil layer) within box.\n"
            )
            return

        for var_name in target_vars:
            changed, total = update_var_to_wilt(nc, var_name, region_land_mask, wilt3d, verbose)
            if verbose:
                sys.stderr.write(
                    f"  {var_name}: set {changed} values across all times/layers (target {total}).\n"
                )


def main() -> None:
    args = parse_args()
    files = expand_files(args.files)
    if not files:
        sys.stderr.write("No files matched the given paths/patterns.\n")
        sys.exit(1)

    error_count = 0
    for path in files:
        try:
            process_file(
                path=path,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
                do_backup=args.backup,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
        except Exception as exc:
            error_count += 1
            sys.stderr.write(f"Error processing {path}: {exc}\n")

    if error_count:
        sys.stderr.write(f"Completed with {error_count} error(s).\n")
        sys.exit(2)


if __name__ == "__main__":
    main()

