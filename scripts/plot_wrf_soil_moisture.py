#!/usr/bin/env python3
"""
Plot soil moisture from a WRF wrfout file as a color map.

Features:
- Auto-detect common soil moisture variables (SMOIS, SH2O, SMCREL, SM)
- Select variable via --var
- Select time index via --time-index and soil layer via --layer-index
- Automatic lat/lon detection (XLAT/XLONG) for georeferenced plotting
- Cartopy-based map with coastlines/borders/states; falls back with clear error if Cartopy missing
- Configurable colormap, vmin/vmax, output path, and extent

Example:
python plot_wrf_soil_moisture.py \
  --file /HDD_Pool/shanru/s2s/Exps/Extremes/wrfsm1_yr15/wrfout_d01_2015-05-01_00:00:00 \
  --var SMOIS --layer-index 0 --time-index 0 --output wrf_smois.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read soil moisture from WRF wrfout and plot a color map."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to WRF wrfout file (netCDF)",
    )
    parser.add_argument(
        "--var",
        default=None,
        help=(
            "Variable name to plot (default: auto-detect among SMOIS, SH2O, SMCREL, SM, SMC)."
        ),
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to select (default: 0).",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Soil layer index to select if applicable (default: 0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsample stride for faster plotting (default: 1 = no subsample).",
    )
    parser.add_argument(
        "--cmap",
        default="YlGnBu",
        help="Matplotlib colormap name (default: YlGnBu).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Color map minimum (default: data min).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color map maximum (default: data max).",
    )
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Map extent as lon_min lon_max lat_min lat_max (default: from data).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <var>_t<time>_l<layer>.png in CWD).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for saved image (default: 150).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title (default: auto).",
    )
    return parser.parse_args()


def ensure_file_exists(path: str) -> None:
    if not os.path.isfile(path):
        sys.stderr.write(f"Error: file not found: {path}\n")
        sys.exit(1)


def open_dataset(path: str) -> xr.Dataset:
    import xarray as xr
    # Use engine auto-detection; disable chunking to avoid dask requirement for simple runs
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception:
        # Retry with netcdf4 engine explicitly
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    return ds


def autodetect_variable(ds: xr.Dataset, user_var: Optional[str]) -> str:
    if user_var:
        if user_var in ds.variables:
            return user_var
        # Some files may store in uppercase; try case-insensitive match
        for v in ds.variables:
            if v.lower() == user_var.lower():
                return v
        available = ", ".join(list(ds.variables))
        raise KeyError(
            f"Variable '{user_var}' not found. Available variables: {available}"
        )

    candidates: List[str] = ["SMOIS", "SH2O", "SMCREL", "SM", "SMC"]
    for name in candidates:
        if name in ds.variables:
            return name
        # case-insensitive
        for v in ds.variables:
            if v.lower() == name.lower():
                return v
    available = ", ".join(list(ds.variables))
    raise KeyError(
        f"Could not auto-detect soil moisture variable. Available variables: {available}"
    )


def find_time_dim(var: xr.DataArray) -> Optional[str]:
    for dim in var.dims:
        dl = dim.lower()
        if dl in ("time", "times", "xtime"):
            return dim
    # Some WRF variables have 'Time' capitalized
    if "Time" in var.dims:
        return "Time"
    return None


def find_layer_dim(var: xr.DataArray) -> Optional[str]:
    for dim in var.dims:
        dl = dim.lower()
        if "soil" in dl or "layer" in dl or "depth" in dl:
            return dim
    # Common WRF name
    if "soil_layers_stag" in var.dims:
        return "soil_layers_stag"
    return None


def get_latlon(ds: xr.Dataset, time_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    import numpy as np
    # Prefer XLAT/XLONG
    lat_name = None
    lon_name = None
    for cand in ("XLAT", "xlat", "lat", "XLAT_M", "CLAT"):
        if cand in ds.variables:
            lat_name = cand
            break
    for cand in ("XLONG", "xlong", "lon", "XLONG_M", "CLONG"):
        if cand in ds.variables:
            lon_name = cand
            break
    if lat_name is None or lon_name is None:
        raise KeyError("Could not find latitude/longitude variables (e.g., XLAT/XLONG).")

    lat = ds[lat_name]
    lon = ds[lon_name]

    # Drop time dimension if present
    if "Time" in lat.dims or "time" in lat.dims or "Times" in lat.dims:
        lat = lat.isel({lat.dims[0]: time_index})
    if "Time" in lon.dims or "time" in lon.dims or "Times" in lon.dims:
        lon = lon.isel({lon.dims[0]: time_index})

    return np.asarray(lat), np.asarray(lon)


def try_get_wrftime(ds: xr.Dataset, time_index: int, time_dim: Optional[str]) -> Optional[str]:
    # Try WRF 'Times' char array
    if "Times" in ds.variables:
        try:
            times = ds["Times"].values
            # times shape: (Time, DateStrLen)
            if times.ndim == 2:
                tstr = "".join(chr(c) for c in times[time_index])
                return tstr.strip()
        except Exception:
            pass

    # Try coordinate vector if exists
    if time_dim and time_dim in ds.coords:
        try:
            val = ds.coords[time_dim].values[time_index]
            return str(val)
        except Exception:
            pass

    # Try XTIME in minutes since start
    if "XTIME" in ds.variables:
        try:
            xtime = ds["XTIME"].values[time_index]
            return f"XTIME={xtime}"
        except Exception:
            pass

    return None


def prepare_data(
    ds: xr.Dataset, var_name: str, time_index: int, layer_index: int
) -> xr.DataArray:
    var = ds[var_name]

    # Select time if present
    tdim = find_time_dim(var)
    if tdim is not None and tdim in var.dims:
        var = var.isel({tdim: time_index})

    # Select soil layer if present
    ldim = find_layer_dim(var)
    if ldim is not None and ldim in var.dims:
        var = var.isel({ldim: layer_index})

    # If after selections still 3D, try to squeeze
    while var.ndim > 2 and any(dim in var.dims for dim in ("Time", "times", "time")):
        # Defensive squeeze of remaining size-1 time dims
        for dim in list(var.dims):
            if var.sizes.get(dim, 2) == 1:
                var = var.isel({dim: 0})
                break
        else:
            break

    if var.ndim != 2:
        raise ValueError(
            f"Selected variable is not 2D after indexing. Shape={var.shape}, dims={var.dims}"
        )

    return var


def plot_map(
    data2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    extent: Optional[Tuple[float, float, float, float]],
    stride: int,
    dpi: int,
    output_path: str,
) -> None:
    # Lazy import to allow --help without heavy deps present
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Subsample if requested
    if stride > 1:
        data2d = data2d[::stride, ::stride]
        lats = lats[::stride, ::stride]
        lons = lons[::stride, ::stride]

    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = plt.axes(projection=proj)

    # Determine extent
    if extent is None:
        lon_min = float(np.nanmin(lons))
        lon_max = float(np.nanmax(lons))
        lat_min = float(np.nanmin(lats))
        lat_max = float(np.nanmax(lats))
        # Add small buffer
        dlon = max(0.1, (lon_max - lon_min) * 0.05)
        dlat = max(0.1, (lat_max - lat_min) * 0.05)
        extent = (lon_min - dlon, lon_max + dlon, lat_min - dlat, lat_max + dlat)
    ax.set_extent(extent, crs=proj)

    # Draw features
    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    try:
        ax.add_feature(cfeature.STATES, linewidth=0.3)
    except Exception:
        pass

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Plot
    mesh = ax.pcolormesh(
        lons,
        lats,
        data2d,
        transform=proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Soil moisture")

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_file_exists(args.file)

    ds = open_dataset(args.file)

    var_name = autodetect_variable(ds, args.var)
    var = ds[var_name]

    # Prepare data array
    data_da = prepare_data(ds, var_name, args.time_index, args.layer_index)

    # Units for colorbar label
    units = var.attrs.get("units", "") if hasattr(var, "attrs") else ""

    # Lat/Lon arrays (use time index 0 for coordinates if they have time dimension)
    lats, lons = get_latlon(ds, time_index=0)

    # vmin/vmax
    import numpy as np
    vmin = args.vmin if args.vmin is not None else float(np.nanmin(data_da.values))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(data_da.values))

    # Title
    autott = f"{var_name} (t={args.time_index}, layer={args.layer_index})"
    tlabel = try_get_wrftime(ds, args.time_index, find_time_dim(var))
    if tlabel:
        autott += f" | {tlabel}"
    if units:
        autott += f" [{units}]"
    title = args.title or autott

    # Output path
    if args.output is None:
        base = f"{var_name.lower()}_t{args.time_index}_l{args.layer_index}.png"
        output_path = os.path.join(os.getcwd(), base)
    else:
        output_path = args.output

    # Plot
    try:
        plot_map(
            data2d=np.asarray(data_da.values),
            lats=lats,
            lons=lons,
            title=title,
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
            extent=tuple(args.extent) if args.extent else None,
            stride=int(args.stride),
            dpi=int(args.dpi),
            output_path=output_path,
        )
    except ModuleNotFoundError as e:
        missing = str(e).split("No module named ")[-1].strip("'\"")
        sys.stderr.write(
            "\nERROR: Missing dependency for plotting: "
            f"{missing}. Install Cartopy and Matplotlib, e.g.:\n"
            "  pip install matplotlib cartopy\n"
        )
        sys.exit(2)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()