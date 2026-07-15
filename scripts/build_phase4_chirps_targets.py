#!/usr/bin/env python3
"""Build and validate the native-grid CHIRPS contract used by F4/F6/F8.

The official run keeps the full rectangular native CHIRPS grid, attaches an
official-IBGE Brazil fraction/centre mask and writes a Zarr cube.  It never
regrids or interpolates.  Existing artifacts are not silently replaced: pass
``--replace-existing`` to archive the old artifact before promoting a staged
build.  The known legacy parquet can likewise be quarantined only through the
explicit ``--quarantine-invalid-legacy`` option.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.maps.spatial_support import (  # noqa: E402
    build_pixel_membership,
    load_ibge_regions,
)
from nino_brasil.artifacts import sha256_file  # noqa: E402
from nino_brasil.io_utils import write_csv_atomic  # noqa: E402
from nino_brasil.targets.chirps_native import (  # noqa: E402
    BRAZIL_BBOX,
    TARGET_CONTRACT_VERSION,
    add_brazil_mask,
    build_native_weekly_targets,
    canonicalize_chirps_daily,
    native_grid_hash,
    native_pixel_table,
    validate_native_target,
)


RAIN_ROOT = ROOT / "data/processed/zarr/brazil_precipitation"
DAILY_NATIVE = ROOT / "data/processed/zarr/features/chirps_native_daily_brazil_box.zarr"
OUTPUT = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
PIXELS = ROOT / "data/processed/parquet/features/phase4_chirps_native_pixels.csv"
MEMBERSHIP = ROOT / "data/processed/parquet/statistics/phase4_chirps_native_brazil_membership.parquet"
IBGE_REGIONS = ROOT / "data/interim/ibge/BR_Regioes_2024/BR_Regioes_2024.shp"
LEGACY_Z = ROOT / "data/processed/parquet/features/phase4_chirps_weekly_zanom.parquet"
LEGACY_PIXELS = ROOT / "data/processed/parquet/features/phase4_chirps_pixels.csv"
QUARANTINE_ROOT = ROOT / "data/quarantine/phase4_chirps"
WEEKLY_BLOCK_ROOT = ROOT / "data/interim/chirps_weekly_native_blocks"
TARGET_VARIABLE_CONTRACT = (
    ROOT
    / "data/processed/parquet/statistics/phase4_chirps_target_variable_contract.csv"
)
PARTIAL_BLOCK_EXIT_CODE = 75
BLOCK_METHOD_VERSION = "chirps-native-weekly-block-v4-conditional-positive-l1-scale"
PRIMARY_ROBUST_Z_SANITY_LIMIT = 100.0


def validate_canonical_target(
    dataset: xr.Dataset,
    *,
    deep: bool = True,
):
    """Validate every standardized target against one strict sanity limit.

    Contract v4 fixes the R95p/R99p denominator rather than exempting those
    derived targets from validation.  Raw millimetres remain unclipped and are
    audited independently; only an implausibly large standardized value fails
    promotion.
    """

    return validate_native_target(
        dataset,
        maximum_abs_robust_z=PRIMARY_ROBUST_Z_SANITY_LIMIT,
        deep=deep,
    )


def _zarr_data_files(path: Path) -> list[Path]:
    """Files whose bytes define array content/schema, excluding mutable attrs."""

    excluded = {".zattrs", ".zmetadata", ".zgroup"}
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in excluded
    )


def _zarr_state_fingerprint(path: Path) -> str:
    payload = [
        (
            item.relative_to(path).as_posix(),
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in _zarr_data_files(path)
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _zarr_content_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in _zarr_data_files(path):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _pixel_mask_fingerprint(pixels: pd.DataFrame) -> str:
    """Fingerprint the logical native-grid mask across a CSV round trip.

    CHIRPS coordinates and the fractional Brazil mask are stored as float32 in
    the target contract.  ``read_csv`` promotes those values to float64; if we
    format the promoted binary values directly, harmless sub-ULP differences
    change the digest even though the CSV and Zarr masks are identical.  Cast
    the three contract floats back to their authoritative storage precision
    before serialising the logical fingerprint.
    """

    columns = [
        "pixel_id",
        "grid_row",
        "grid_column",
        "lat",
        "lon",
        "brazil_fraction",
        "brazil_center",
        "grid_hash",
    ]
    missing = set(columns).difference(pixels.columns)
    if missing:
        raise KeyError(f"Pixel-mask fingerprint missing columns: {sorted(missing)}")
    canonical = pixels.sort_values("pixel_id", kind="mergesort")[columns].copy()
    canonical[["lat", "lon", "brazil_fraction"]] = canonical[
        ["lat", "lon", "brazil_fraction"]
    ].astype(np.float32)
    payload = canonical.to_csv(
        index=False, float_format="%.10g"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _target_variable_contract(
    dataset: xr.Dataset,
    *,
    build_id: str,
    block_signature: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, variable in dataset.data_vars.items():
        if name.startswith("spi_gamma_") and "weekly_origin" not in name:
            role = "spi_fitted_parameter"
        elif "climatology" in name or name.startswith("baseline_wet_day"):
            role = "auditable_fitted_parameter"
        elif name in {"pixel_id", "brazil_fraction", "brazil_center"}:
            role = "native_grid_identity_or_mask"
        elif "anomaly" in name or "robust_z" in name:
            role = "inferential_target"
        else:
            role = "native_observation_or_diagnostic"
        rows.append(
            {
                "build_id": build_id,
                "variable": name,
                "role": role,
                "dimensions": ";".join(variable.dims),
                "shape": "x".join(str(variable.sizes[dimension]) for dimension in variable.dims),
                "dtype": str(variable.dtype),
                "units": str(variable.attrs.get("units", "")),
                "method": str(
                    variable.attrs.get("method", variable.attrs.get("definition", ""))
                ),
                "source_index": str(variable.attrs.get("source_index", "")),
                "grid_hash_sha256": str(dataset.attrs.get("grid_hash_sha256", "")),
                "target_contract_version": str(
                    dataset.attrs.get("target_contract_version", "")
                ),
                "block_signature_sha256": block_signature,
                "numeric_authority": "Zarr variable on exact native CHIRPS coordinates",
            }
        )
    return pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)


def _archive_existing(path: Path, *, reason: str) -> Path:
    """Move an artifact to a timestamped quarantine; never delete it."""

    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ValueError(f"Refusing to quarantine outside the project: {resolved}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = QUARANTINE_ROOT / stamp / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    manifest = destination.parent / "manifest.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(path.relative_to(ROOT)),
        "destination": str(destination.relative_to(ROOT)),
        "reason": reason,
        "operation": "move; no deletion",
    }
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def audit_legacy_parquet(path: Path = LEGACY_Z) -> dict[str, object]:
    """Detect the historical near-zero-divisor failure without modifying data."""

    report: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return report
    table = pd.read_parquet(path)
    values = table.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    absolute = np.abs(values[finite])
    report.update(
        {
            "shape": list(table.shape),
            "max_abs": float(absolute.max()) if absolute.size else None,
            "fraction_abs_gt_20": float((absolute > 20).mean()) if absolute.size else None,
            "fraction_abs_gt_100": float((absolute > 100).mean()) if absolute.size else None,
            "valid": bool(absolute.size and absolute.max() <= 100),
            "criterion": "finite robust standardized precipitation should have max |z| <= 100",
        }
    )
    return report


def _mask_pixels(
    pixels: pd.DataFrame,
    *,
    centroid_only: bool,
    noncanonical_suffix: str = "",
) -> pd.DataFrame:
    if not IBGE_REGIONS.exists():
        raise FileNotFoundError(f"Official IBGE region shapefile not found: {IBGE_REGIONS}")
    regions = load_ibge_regions(IBGE_REGIONS)
    grid_hash = str(pixels["grid_hash"].iloc[0])
    if noncanonical_suffix:
        cache_path = MEMBERSHIP.with_name(
            MEMBERSHIP.stem + f"_{noncanonical_suffix}.parquet"
        )
    else:
        cache_path = (
            MEMBERSHIP.with_name(MEMBERSHIP.stem + "_centroid_quick.parquet")
            if centroid_only
            else MEMBERSHIP
        )
    if cache_path.exists():
        membership = pd.read_parquet(cache_path)
        if "grid_hash" not in membership or set(membership["grid_hash"].astype(str)) != {grid_hash}:
            raise ValueError(
                f"Membership cache {cache_path} belongs to another grid; archive it explicitly."
            )
    else:
        membership = build_pixel_membership(
            pixels[["pixel_id", "lat", "lon"]],
            regions,
            boundary_method="centroid" if centroid_only else "area",
        )
        membership["grid_hash"] = grid_hash
        membership["mask_contract"] = (
            "quick-centroid; not canonical" if centroid_only else "official IBGE equal-area overlap"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(
            cache_path.suffix + f".tmp-{uuid.uuid4().hex[:8]}"
        )
        membership.to_parquet(temporary, index=False)
        temporary.replace(cache_path)
    fraction = membership.groupby("pixel_id", sort=False)[
        "fracao_pixel_na_unidade"
    ].sum().clip(0.0, 1.0)
    centre = membership.loc[
        membership["centro_na_unidade"].astype(bool), "pixel_id"
    ].drop_duplicates()
    out = pixels.copy()
    out["brazil_fraction"] = out["pixel_id"].map(fraction).fillna(0.0).astype("float32")
    out["brazil_center"] = out["pixel_id"].isin(set(centre)).astype(bool)
    out["brazil_mask_method"] = (
        "centroid_quick" if centroid_only else "IBGE_regions_2024_equal_area_fraction"
    )
    return out


def _promote_staged(staged: Path, destination: Path, *, replace_existing: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not replace_existing:
            raise FileExistsError(
                f"{destination} already exists. Validate it or pass --replace-existing "
                "to archive it before promotion."
            )
        archived = _archive_existing(destination, reason="replaced by validated native CHIRPS target")
        print(f"[archive] {archived.relative_to(ROOT)}")
    staged.replace(destination)


def _write_pixels(pixels: pd.DataFrame, *, replace_existing: bool) -> None:
    PIXELS.parent.mkdir(parents=True, exist_ok=True)
    if PIXELS.exists():
        old = pd.read_csv(PIXELS)
        same_inventory = False
        try:
            same_inventory = _pixel_mask_fingerprint(old) == _pixel_mask_fingerprint(
                pixels
            )
        except (KeyError, ValueError):
            same_inventory = False
        if same_inventory and not replace_existing:
            return
        if not replace_existing:
            raise FileExistsError(f"{PIXELS} exists for a different/replaceable build.")
        archived = _archive_existing(PIXELS, reason="replaced by native target pixel inventory")
        print(f"[archive] {archived.relative_to(ROOT)}")
    temporary = PIXELS.with_suffix(PIXELS.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
    pixels.to_csv(temporary, index=False)
    written = pd.read_csv(temporary)
    required = {"pixel_id", "grid_row", "grid_column", "lat", "lon", "grid_hash"}
    if required.difference(written.columns) or len(written) != len(pixels):
        raise ValueError("Staged native pixel inventory failed schema/row validation.")
    temporary.replace(PIXELS)


def _daily_source_time_contract(
    stores: list[Path],
) -> tuple[pd.DatetimeIndex, list[str], str]:
    times: list[pd.DatetimeIndex] = []
    names: list[str] = []
    reference_grid: str | None = None
    for store in stores:
        source = xr.open_zarr(store, consolidated=None)
        daily = canonicalize_chirps_daily(source, variable="precip", bbox=BRAZIL_BBOX)
        grid = native_grid_hash(daily.latitude.values, daily.longitude.values)
        if reference_grid is None:
            reference_grid = grid
        elif reference_grid != grid:
            raise ValueError(f"Native CHIRPS grid changed at {store}.")
        index = pd.DatetimeIndex(daily.time.values)
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"Invalid timestamps in source store {store}.")
        times.append(index)
        names.append(store.name)
        source.close()
    combined = pd.DatetimeIndex(np.concatenate([index.values for index in times]))
    if combined.has_duplicates or not combined.is_monotonic_increasing:
        raise ValueError("CHIRPS source stores overlap or are out of chronological order.")
    if len(combined) > 1 and not bool(
        (np.diff(combined.values) == np.timedelta64(1, "D")).all()
    ):
        raise ValueError("CHIRPS source stores do not form a continuous daily calendar.")
    return combined, names, str(reference_grid or "")


def _finalize_daily_lineage(
    destination: Path,
    *,
    stores: list[Path],
) -> xr.DataArray:
    expected_index, source_names, source_grid = _daily_source_time_contract(stores)
    dataset = xr.open_zarr(destination, consolidated=False)
    daily = dataset["precip_daily_mm"]
    actual_index = pd.DatetimeIndex(daily.time.values)
    actual_grid = native_grid_hash(daily.latitude.values, daily.longitude.values)
    if not actual_index.equals(expected_index):
        raise ValueError(
            f"Daily native cache {destination} does not exactly match all source dates."
        )
    if actual_grid != source_grid:
        raise ValueError(f"Daily native cache {destination} differs from source grid.")
    existing_attrs = dict(dataset.attrs)
    dataset.close()
    state_fingerprint = _zarr_state_fingerprint(destination)
    if existing_attrs.get("daily_data_state_sha256") == state_fingerprint:
        content_fingerprint = str(
            existing_attrs.get("daily_data_content_sha256", "")
        )
    else:
        content_fingerprint = ""
    if not content_fingerprint:
        print("[daily native] hashing immutable Zarr array content for lineage", flush=True)
        content_fingerprint = _zarr_content_fingerprint(destination)
    source_names_sha256 = hashlib.sha256(
        json.dumps(source_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_attrs = {
        "grid_hash_sha256": actual_grid,
        "completed_source_stores_json": json.dumps(source_names),
        "source_store_count": len(source_names),
        "source_store_names_sha256": source_names_sha256,
        "daily_data_state_sha256": state_fingerprint,
        "daily_data_content_sha256": content_fingerprint,
        "daily_count": len(actual_index),
        "daily_start": actual_index.min().date().isoformat(),
        "daily_end": actual_index.max().date().isoformat(),
        "daily_calendar_continuous": True,
        "spatial_operation": "native coordinate slice; interpolation=false",
    }
    if any(existing_attrs.get(key) != value for key, value in expected_attrs.items()):
        import zarr

        for attempt in range(5):
            try:
                group = zarr.open_group(str(destination), mode="a")
                group.attrs.update(expected_attrs)
                zarr.consolidate_metadata(str(destination))
                break
            except PermissionError:
                if attempt == 4:
                    raise
                # Windows can briefly retain a metadata handle after xarray closes.
                time.sleep(0.25 * (attempt + 1))
    else:
        print("[daily native] immutable lineage already current; metadata write skipped")
    reopened = xr.open_zarr(destination, consolidated=True)
    output = reopened["precip_daily_mm"]
    output.attrs.update(
        {
            "daily_data_content_sha256": content_fingerprint,
            "daily_data_state_sha256": state_fingerprint,
            "source_store_count": len(source_names),
            "source_store_names_sha256": source_names_sha256,
        }
    )
    return output


def materialize_daily_native(
    stores: list[Path],
    destination: Path,
    *,
    rebuild: bool,
    resume_staging: Path | None = None,
    resume_through_date: str | None = None,
) -> xr.DataArray:
    """Append annual native daily slices to one persistent pre-resample cube.

    Materialising the concatenation is both the scientific fix for the former
    year-boundary averaging and a memory-control boundary.  It avoids a Dask
    graph spanning the large original 200x200 source chunks for every robust
    climatology operation.
    """

    if destination.exists() and not rebuild:
        dataset = xr.open_zarr(destination, consolidated=None)
        if "precip_daily_mm" not in dataset:
            raise ValueError(f"Daily native cache {destination} has no precip_daily_mm.")
        daily = dataset["precip_daily_mm"]
        grid = native_grid_hash(daily.latitude.values, daily.longitude.values)
        if str(dataset.attrs.get("grid_hash_sha256", "")) != grid:
            raise ValueError(f"Daily native cache {destination} has a grid-hash mismatch.")
        index = pd.DatetimeIndex(daily.time.values)
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"Daily native cache {destination} has invalid timestamps.")
        if len(index) > 1 and not bool(
            (np.diff(index.values) == np.timedelta64(1, "D")).all()
        ):
            raise ValueError(f"Daily native cache {destination} has calendar gaps.")
        dataset.close()
        return _finalize_daily_lineage(destination, stores=stores)
    if destination.exists():
        archived = _archive_existing(
            destination,
            reason="explicit --rebuild-daily before native daily concatenation",
        )
        print(f"[archive] {archived.relative_to(ROOT)}")

    staged = (
        resume_staging
        if resume_staging is not None
        else destination.with_name(destination.name + f".staging-{uuid.uuid4().hex[:8]}")
    )
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    previous_last: pd.Timestamp | None = None
    first = True
    if resume_staging is not None:
        staged = staged.resolve()
        if ROOT.resolve() not in staged.parents or not staged.is_dir():
            raise ValueError(f"Invalid --resume-daily-staging path: {staged}")
        staged_dataset = xr.open_zarr(staged, consolidated=False)
        if "precip_daily_mm" not in staged_dataset:
            raise ValueError(f"Resume staging has no precip_daily_mm: {staged}")
        staged_daily = staged_dataset["precip_daily_mm"]
        reference_lat = np.asarray(staged_daily.latitude.values)
        reference_lon = np.asarray(staged_daily.longitude.values)
        staged_index = pd.DatetimeIndex(staged_daily.time.values)
        if staged_index.has_duplicates or not staged_index.is_monotonic_increasing:
            raise ValueError(f"Resume staging has invalid timestamps: {staged}")
        import zarr

        resume_group = zarr.open_group(str(staged), mode="a")
        recorded_checkpoint = resume_group.attrs.get("completed_through_date")
        checkpoint = resume_through_date or recorded_checkpoint
        if checkpoint is None:
            raise ValueError(
                "Interrupted staging has no completed_through_date. Supply "
                "--resume-through-date with the last verified complete date."
            )
        checkpoint_time = pd.Timestamp(str(checkpoint))
        keep = int((staged_index <= checkpoint_time).sum())
        if keep == 0 or staged_index[keep - 1] != checkpoint_time:
            raise ValueError(
                f"Checkpoint {checkpoint_time.date()} is not an exact staged daily timestamp."
            )
        if keep < len(staged_index):
            resume_group["precip_daily_mm"].resize(
                (keep, len(reference_lat), len(reference_lon))
            )
            resume_group["time"].resize((keep,))
            staged_dataset = xr.open_zarr(staged, consolidated=False)
            staged_daily = staged_dataset["precip_daily_mm"]
            staged_index = pd.DatetimeIndex(staged_daily.time.values)
            print(
                f"[resume] truncated uncheckpointed tail to {checkpoint_time.date()}",
                flush=True,
            )
        resume_group.attrs["completed_through_date"] = checkpoint_time.date().isoformat()
        previous_last = checkpoint_time
        first = False
        print(
            f"[resume] {staged.relative_to(ROOT)} through {previous_last.date()} "
            f"({len(staged_index)} days)",
            flush=True,
        )
    for store in stores:
        source = xr.open_zarr(store, consolidated=None)
        daily = canonicalize_chirps_daily(source, variable="precip", bbox=BRAZIL_BBOX)
        latitude = np.asarray(daily.latitude.values)
        longitude = np.asarray(daily.longitude.values)
        if reference_lat is None:
            reference_lat, reference_lon = latitude, longitude
        elif not (
            np.array_equal(reference_lat, latitude)
            and np.array_equal(reference_lon, longitude)
        ):
            raise ValueError(f"Native CHIRPS grid changed at {store}; refusing concatenation.")
        index = pd.DatetimeIndex(daily.time.values)
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"Invalid daily timestamps in {store}.")
        if previous_last is not None and index.max() <= previous_last:
            print(f"  [daily native] skip {store.name}; already staged", flush=True)
            continue
        if previous_last is not None and index.min() <= previous_last:
            raise ValueError(
                f"Annual CHIRPS stores overlap at {store}; duplicate dates are not averaged."
            )
        previous_last = index.max()
        dataset = daily.to_dataset(name="precip_daily_mm")
        # Source Zarr encoding may carry a v3 ``serializer`` object.  Encoding
        # is storage-specific lineage, not scientific metadata, and cannot be
        # copied into the canonical v2 store.  Re-declare only our chunk plan.
        for variable in dataset.variables:
            dataset[variable].encoding = {}
        dataset = dataset.chunk(
            {
                "time": min(31, dataset.sizes["time"]),
                "latitude": min(42, dataset.sizes["latitude"]),
                "longitude": min(43, dataset.sizes["longitude"]),
            }
        )
        dataset.attrs.update(
            {
                "grid_hash_sha256": native_grid_hash(latitude, longitude),
                "spatial_operation": "native coordinate slice; interpolation=false",
                "temporal_contract": "annual daily stores appended before any weekly resample",
            }
        )
        if first:
            dataset.to_zarr(
                staged,
                mode="w",
                consolidated=False,
                zarr_format=2,
                encoding={"precip_daily_mm": {"chunks": (31, 42, 43)}},
            )
            import zarr

            first_group = zarr.open_group(str(staged), mode="a")
            first_group.attrs["completed_through_date"] = index.max().date().isoformat()
            first_group.attrs["completed_source_stores_json"] = json.dumps([store.name])
            first = False
        else:
            # Xarray/Dask append may launch several Windows writers against the
            # same partially filled 31-day Zarr chunk.  A single direct Zarr
            # assignment is deterministic and avoids WinError 5 races.
            import zarr

            group = zarr.open_group(str(staged), mode="a")
            precip_array = group["precip_daily_mm"]
            time_array = group["time"]
            old_size = int(precip_array.shape[0])
            new_size = old_size + len(index)
            precip_array.resize((new_size, len(latitude), len(longitude)))
            time_array.resize((new_size,))
            units = str(time_array.attrs.get("units", ""))
            if not units.startswith("days since "):
                raise ValueError(f"Unsupported staged time encoding: {units!r}")
            origin = pd.Timestamp(units.removeprefix("days since "))
            encoded_time = ((index - origin) / pd.Timedelta(days=1)).to_numpy(dtype=np.int64)
            try:
                precip_array[old_size:new_size, :, :] = np.asarray(
                    daily.values, dtype=np.float32
                )
                time_array[old_size:new_size] = encoded_time
            except Exception:
                # Shapes may have grown, but the checkpoint is intentionally not
                # advanced. A subsequent resume truncates this unverified tail.
                raise
            completed = json.loads(group.attrs.get("completed_source_stores_json", "[]"))
            completed.append(store.name)
            group.attrs["completed_source_stores_json"] = json.dumps(completed)
            group.attrs["completed_through_date"] = index.max().date().isoformat()
        print(
            f"  [daily native] {store.name}: {index.min().date()}..{index.max().date()}",
            flush=True,
        )
    staged_dataset = xr.open_zarr(staged, consolidated=False)
    staged_index = pd.DatetimeIndex(staged_dataset.time.values)
    if staged_index.has_duplicates or not staged_index.is_monotonic_increasing:
        raise ValueError(f"Staged daily concatenation is invalid and remains at {staged}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(destination)
    import zarr

    zarr.consolidate_metadata(str(destination))
    return _finalize_daily_lineage(destination, stores=stores)


def _weekly_block_signature(
    daily: xr.DataArray,
    pixels: pd.DataFrame,
    *,
    include_spi: bool,
    include_extremes: bool,
    latitude_block_size: int,
    build_status: str,
) -> tuple[str, dict[str, object]]:
    index = pd.DatetimeIndex(daily.time.values)
    payload: dict[str, object] = {
        "contract": "chirps-native-weekly-spatial-blocks-v4",
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "method_version": BLOCK_METHOD_VERSION,
        "full_grid_hash_sha256": native_grid_hash(
            daily.latitude.values, daily.longitude.values
        ),
        "daily_start": index.min().isoformat(),
        "daily_end": index.max().isoformat(),
        "daily_count": len(index),
        "latitude_count": int(daily.sizes["latitude"]),
        "longitude_count": int(daily.sizes["longitude"]),
        "latitude_block_size": int(latitude_block_size),
        "include_spi": bool(include_spi),
        "include_extremes": bool(include_extremes),
        "minimum_valid_days": 7,
        "climatology_baseline": "1991-01-01/2020-12-31",
        "minimum_baseline_years": 20,
        "wet_day_threshold_mm": 1.0,
        "minimum_baseline_wet_days": 100,
        "precip_absolute_scale_floor": 0.10,
        "spell_absolute_scale_floor_days": 0.25,
        "robust_scale_candidates": [
            "seasonal_1.4826_mad",
            "pooled_residual_1.4826_mad",
            "pooled_residual_sqrt_pi_over_2_mean_absolute",
            "r95p_r99p_positive_week_sqrt_pi_over_2_mean_absolute",
            "r95p_r99p_wet_day_percentile_threshold_floor_when_positive_support_insufficient",
        ],
        "zero_inflated_tail_minimum_positive_weeks": 20,
        "zero_inflated_tail_fallback": (
            "audited pixelwise baseline wet-day p95/p99 threshold floor; "
            "no imputation or spatial pooling"
        ),
        "robust_scale_selection": (
            "largest supported candidate; R95p/R99p conditional-positive "
            "magnitude scale prevents dilution by zero occurrence probability; "
            "reference climatology only"
        ),
        "daily_data_content_sha256": str(
            daily.attrs.get("daily_data_content_sha256", "")
        ),
        "pixel_mask_sha256": _pixel_mask_fingerprint(pixels),
        "mask_method": str(pixels["brazil_mask_method"].iloc[0]),
        "build_status": build_status,
    }
    if not payload["daily_data_content_sha256"]:
        # DataArray attrs may not inherit root-group attrs in every xarray
        # version. Read them from the opened store's encoding source when
        # available; absence is a hard provenance failure.
        source_path = daily.encoding.get("source")
        if source_path:
            source_dataset = xr.open_zarr(source_path, consolidated=True)
            payload["daily_data_content_sha256"] = str(
                source_dataset.attrs.get("daily_data_content_sha256", "")
            )
            source_dataset.close()
    if not payload["daily_data_content_sha256"]:
        raise ValueError("Daily native content fingerprint is missing.")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _drop_cli_options(
    arguments: list[str],
    *,
    flags: set[str] = frozenset(),
    valued: set[str] = frozenset(),
) -> list[str]:
    cleaned: list[str] = []
    position = 0
    while position < len(arguments):
        value = arguments[position]
        if value in flags:
            position += 1
            continue
        if value in valued:
            position += 2
            continue
        cleaned.append(value)
        position += 1
    return cleaned


def materialize_weekly_native_blocks(
    daily: xr.DataArray,
    pixels: pd.DataFrame,
    *,
    include_spi: bool,
    include_extremes: bool,
    latitude_block_size: int,
    build_status: str,
    max_new_blocks: int | None = None,
    only_block_start: int | None = None,
) -> tuple[xr.Dataset | None, Path]:
    """Compute F4 targets in restartable latitude stripes.

    Robust climatologies require the complete time series for every pixel, but
    pixels are statistically separable during target construction.  Persisting
    a validated latitude stripe at a time bounds both the Dask graph and peak
    RAM without averaging, interpolating or otherwise changing a CHIRPS cell.
    """

    if latitude_block_size < 1:
        raise ValueError("--latitude-block-size must be positive.")
    signature, signature_payload = _weekly_block_signature(
        daily,
        pixels,
        include_spi=include_spi,
        include_extremes=include_extremes,
        latitude_block_size=latitude_block_size,
        build_status=build_status,
    )
    block_root = WEEKLY_BLOCK_ROOT / signature[:16]
    manifest_path = block_root / "manifest.json"
    expected_manifest: dict[str, object] = {
        **signature_payload,
        "signature_sha256": signature,
        "operation": "independent native-latitude stripes; no regridding",
        "completed_blocks": [],
        "block_records": {},
    }
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded.get("signature_sha256") != signature:
            raise ValueError(f"Weekly block manifest signature mismatch: {manifest_path}")
        expected_manifest["completed_blocks"] = list(recorded.get("completed_blocks", []))
        expected_manifest["block_records"] = dict(recorded.get("block_records", {}))
    else:
        _atomic_json(manifest_path, expected_manifest)

    if max_new_blocks is not None and max_new_blocks < 1:
        raise ValueError("max_new_blocks must be positive when provided.")

    block_paths: list[Path] = []
    latitude_count = int(daily.sizes["latitude"])
    total_blocks = len(range(0, latitude_count, latitude_block_size))
    new_blocks = 0
    reused_blocks = 0
    for start in range(0, latitude_count, latitude_block_size):
        stop = min(start + latitude_block_size, latitude_count)
        name = f"latitude_{start:03d}_{stop:03d}.zarr"
        block_path = block_root / name
        block_paths.append(block_path)
        if only_block_start is not None and start != only_block_start:
            continue
        reusable = False
        if block_path.exists():
            try:
                existing = xr.open_zarr(block_path, consolidated=True)
                validation = validate_canonical_target(existing, deep=False)
                reusable = bool(
                    validation.valid
                    and existing.attrs.get("block_signature_sha256") == signature
                    and int(existing.attrs.get("block_latitude_start", -1)) == start
                    and int(existing.attrs.get("block_latitude_stop", -1)) == stop
                    and np.array_equal(
                        existing.latitude.values, daily.latitude.isel(latitude=slice(start, stop)).values
                    )
                )
                existing.close()
                record = dict(expected_manifest["block_records"]).get(name, {})
                if reusable:
                    state_sha = _zarr_state_fingerprint(block_path)
                    if record.get("data_state_sha256") == state_sha:
                        reusable = bool(record.get("data_content_sha256"))
                    else:
                        content_sha = _zarr_content_fingerprint(block_path)
                        reusable = bool(
                            record.get("data_content_sha256") == content_sha
                        )
                        if reusable and only_block_start is None:
                            record["data_state_sha256"] = state_sha
                            records = dict(expected_manifest["block_records"])
                            records[name] = record
                            expected_manifest["block_records"] = records
            except Exception:
                reusable = False
            if not reusable:
                archived = _archive_existing(
                    block_path,
                    reason="invalid or mismatched restartable CHIRPS weekly latitude block",
                )
                print(f"[archive] {archived.relative_to(ROOT)}", flush=True)
        if reusable:
            reused_blocks += 1
            if only_block_start is None:
                completed = list(expected_manifest["completed_blocks"])
                if name not in completed:
                    completed.append(name)
                    expected_manifest["completed_blocks"] = sorted(completed)
                    _atomic_json(manifest_path, expected_manifest)
            if max_new_blocks is None:
                print(f"  [weekly native] reuse {name}", flush=True)
            continue

        if max_new_blocks is not None and new_blocks >= max_new_blocks:
            completed = set(expected_manifest["completed_blocks"])
            expected_manifest["build_complete"] = False
            expected_manifest["remaining_blocks"] = total_blocks - len(completed)
            expected_manifest["worker_memory_boundary"] = (
                f"at most {max_new_blocks} new latitude block(s) per process"
            )
            _atomic_json(manifest_path, expected_manifest)
            print(
                "[weekly native] worker boundary reached; "
                f"{expected_manifest['remaining_blocks']} block(s) remain",
                flush=True,
            )
            return None, manifest_path

        print(
            f"  [weekly native] resume {reused_blocks}/{total_blocks} reusable; "
            f"build rows {start}:{stop} "
            f"of {latitude_count} ({name})",
            flush=True,
        )
        # Four latitude rows are small enough to materialize eagerly (~45 MB
        # daily). This removes a very large repeated Dask groupby/quantile graph
        # while preserving every native value and bounds peak RAM.
        daily_block = daily.isel(latitude=slice(start, stop)).load()
        pixel_block = pixels.loc[
            (pixels["grid_row"] >= start) & (pixels["grid_row"] < stop)
        ].copy()
        target_block = build_native_weekly_targets(
            daily_block,
            pixels=pixel_block,
            minimum_valid_days=7,
            include_spi=include_spi,
            include_extremes=include_extremes,
        )
        target_block.attrs.update(
            {
                "build_status": build_status,
                "block_signature_sha256": signature,
                "block_full_grid_hash_sha256": signature_payload["full_grid_hash_sha256"],
                "block_latitude_start": start,
                "block_latitude_stop": stop,
            }
        )
        validation = validate_canonical_target(target_block, deep=False)
        if not validation.valid:
            raise ValueError(f"Latitude block {name} failed validation: {validation.errors}")
        staged = block_path.with_name(block_path.name + ".staging")
        if staged.exists():
            archived = _archive_existing(
                staged,
                reason="incomplete previous CHIRPS weekly latitude-block staging",
            )
            print(f"[archive] {archived.relative_to(ROOT)}", flush=True)
        chunks = {
            "time": min(52, target_block.sizes["time"]),
            "latitude": stop - start,
            "longitude": min(43, target_block.sizes["longitude"]),
        }
        for variable in target_block.variables:
            target_block[variable].encoding = {}
        for attempt in range(1, 4):
            try:
                target_block.chunk(chunks).to_zarr(
                    staged, mode="w", consolidated=True, zarr_format=2
                )
                break
            except PermissionError:
                if staged.exists():
                    archived = _archive_existing(
                        staged,
                        reason=(
                            "transient Windows metadata lock while writing "
                            f"CHIRPS block; attempt {attempt}"
                        ),
                    )
                    print(f"[archive] {archived.relative_to(ROOT)}", flush=True)
                if attempt == 3:
                    raise
        staged_dataset = xr.open_zarr(staged, consolidated=True)
        staged_validation = validate_canonical_target(staged_dataset, deep=False)
        staged_dataset.close()
        if not staged_validation.valid:
            raise ValueError(
                f"Staged latitude block {name} failed validation: "
                f"{staged_validation.errors}"
            )
        staged.replace(block_path)
        block_record = {
            "latitude_start": start,
            "latitude_stop": stop,
            "data_state_sha256": _zarr_state_fingerprint(block_path),
            "data_content_sha256": _zarr_content_fingerprint(block_path),
            "validated": staged_validation.as_dict(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        if only_block_start is None:
            completed = list(expected_manifest["completed_blocks"])
            if name not in completed:
                completed.append(name)
            expected_manifest["completed_blocks"] = sorted(completed)
            records = dict(expected_manifest["block_records"])
            records[name] = block_record
            expected_manifest["block_records"] = records
            expected_manifest["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(manifest_path, expected_manifest)
        new_blocks += 1
        del target_block, daily_block
        gc.collect()

    if only_block_start is not None:
        print(
            f"[weekly native] isolated block {only_block_start} finished; "
            "parent will reconcile the manifest",
            flush=True,
        )
        return None, manifest_path

    expected_manifest["blocks_complete"] = True
    expected_manifest["build_complete"] = False
    expected_manifest["remaining_blocks"] = 0
    expected_manifest["blocks_completed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, expected_manifest)
    opened = [xr.open_zarr(path, consolidated=True) for path in block_paths]
    try:
        combined = xr.concat(
            opened,
            dim="latitude",
            data_vars="minimal",
            coords="minimal",
            compat="equals",
            join="exact",
            combine_attrs="override",
        )
        if not np.array_equal(combined.latitude.values, daily.latitude.values):
            raise ValueError("Combined latitude blocks do not reproduce the native grid.")
        combined.attrs.update(
            {
                "grid_hash_sha256": signature_payload["full_grid_hash_sha256"],
                "build_status": build_status,
                "block_signature_sha256": signature,
                "block_latitude_start": 0,
                "block_latitude_stop": latitude_count,
                "block_materialization": (
                    "restartable latitude stripes concatenated by exact coordinates; "
                    "pixel values unchanged"
                ),
            }
        )
        validation = validate_canonical_target(combined, deep=False)
        if not validation.valid:
            raise ValueError(f"Combined weekly blocks failed validation: {validation.errors}")
        return combined, manifest_path
    except Exception:
        for dataset in opened:
            dataset.close()
        raise


def build(args: argparse.Namespace) -> int:
    stores = sorted(RAIN_ROOT.glob("chirps_p25_*.zarr"))
    if args.start_year is not None:
        stores = [p for p in stores if int(p.stem.rsplit("_", 1)[-1]) >= args.start_year]
    if args.end_year is not None:
        stores = [p for p in stores if int(p.stem.rsplit("_", 1)[-1]) <= args.end_year]
    if not stores:
        raise FileNotFoundError(f"No CHIRPS p25 annual stores in {RAIN_ROOT}")
    print(f"[CHIRPS] {len(stores)} daily stores; concatenate before weekly resampling")
    noncanonical = args.start_year is not None or args.end_year is not None
    if noncanonical:
        first_year = args.start_year or int(stores[0].stem.rsplit("_", 1)[-1])
        last_year = args.end_year or int(stores[-1].stem.rsplit("_", 1)[-1])
        daily_path = DAILY_NATIVE.with_name(
            f"chirps_native_daily_brazil_box_quick_{first_year}_{last_year}.zarr"
        )
        noncanonical_suffix = f"quick_{first_year}_{last_year}"
    else:
        daily_path = DAILY_NATIVE
        noncanonical_suffix = ""
    daily = materialize_daily_native(
        stores,
        daily_path,
        rebuild=args.rebuild_daily,
        resume_staging=args.resume_daily_staging,
        resume_through_date=args.resume_through_date,
    )
    pixels = native_pixel_table(daily.latitude.values, daily.longitude.values)
    pixels = _mask_pixels(
        pixels,
        centroid_only=args.centroid_mask_quick,
        noncanonical_suffix=noncanonical_suffix,
    )
    if args.centroid_mask_quick or noncanonical:
        build_status = (
            "noncanonical subset/quick mask; must not replace canonical official target"
        )
    else:
        build_status = "canonical"
    if noncanonical:
        destination = OUTPUT.with_name(
            OUTPUT.stem + f"_{noncanonical_suffix}.zarr"
        )
    elif args.centroid_mask_quick:
        destination = OUTPUT.with_name(OUTPUT.stem + "_centroid_quick.zarr")
    else:
        destination = OUTPUT
    target, block_manifest = materialize_weekly_native_blocks(
        daily,
        pixels,
        include_spi=not args.skip_spi,
        include_extremes=not args.skip_extremes,
        latitude_block_size=args.latitude_block_size,
        build_status=build_status,
        max_new_blocks=args.max_new_weekly_blocks,
        only_block_start=args.only_weekly_block_start,
    )
    if target is None:
        print(f"[restart manifest] {block_manifest.relative_to(ROOT)}")
        return PARTIAL_BLOCK_EXIT_CODE
    build_id = (
        "F4TARGET_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + str(target.attrs.get("block_signature_sha256", ""))[:10]
    )
    target.attrs["build_id"] = build_id
    validation = validate_canonical_target(target, deep=False)
    print(json.dumps(validation.as_dict(), indent=2, ensure_ascii=False))
    if not validation.valid:
        raise ValueError("Native CHIRPS target failed validation; nothing will be promoted.")
    staged = destination.with_name(destination.name + f".staging-{uuid.uuid4().hex[:8]}")
    chunks = {
        "time": min(52, target.sizes["time"]),
        "latitude": min(42, target.sizes["latitude"]),
        "longitude": min(43, target.sizes["longitude"]),
    }
    # ``target`` is concatenated from restartable four-row Zarr blocks.  Xarray
    # preserves each source array's on-disk ``encoding['chunks']`` metadata,
    # even after the Dask array is rechunked below.  Keeping that inherited
    # (52, 4, 43) encoding would make the final 42-row write overlap Dask
    # chunks and xarray correctly refuses it as potentially corrupt.  The
    # canonical store must derive fresh encoding from the explicit final chunk
    # plan, just as each latitude block does when it is first materialised.
    for variable in target.variables:
        target[variable].encoding = {}
    target.chunk(chunks).to_zarr(staged, mode="w", consolidated=True, zarr_format=2)
    staged_dataset = xr.open_zarr(staged, consolidated=True)
    try:
        staged_validation = validate_canonical_target(staged_dataset)
    finally:
        staged_dataset.close()
    if not staged_validation.valid:
        raise ValueError(
            f"Staged target failed validation and was left at {staged}: "
            f"{staged_validation.errors}"
        )
    import zarr

    root_group = zarr.open_group(str(staged), mode="a")
    root_group.attrs["deep_validation_passed"] = True
    root_group.attrs["deep_validation_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    root_group.attrs["deep_validation_primary_max_abs_z_threshold"] = (
        PRIMARY_ROBUST_Z_SANITY_LIMIT
    )
    root_group.attrs["zero_inflated_extreme_policy"] = (
        "R95p/R99p raw millimetres remain unclipped; their derived z uses a "
        "positive-week conditional L1 scale with an audited wet-day threshold "
        "fallback, and obeys the same sanity limit as every robust target"
    )
    zarr.consolidate_metadata(str(staged))
    _promote_staged(staged, destination, replace_existing=args.replace_existing)
    if not noncanonical and not args.centroid_mask_quick:
        _write_pixels(pixels, replace_existing=args.replace_existing)
        written_pixels = PIXELS
    else:
        quick_pixels = PIXELS.with_name(
            PIXELS.stem + f"_{noncanonical_suffix or 'centroid_quick'}.csv"
        )
        quick_pixels.parent.mkdir(parents=True, exist_ok=True)
        quick_temporary = quick_pixels.with_suffix(
            quick_pixels.suffix + f".tmp-{uuid.uuid4().hex[:8]}"
        )
        pixels.to_csv(quick_temporary, index=False)
        quick_temporary.replace(quick_pixels)
        written_pixels = quick_pixels
    variable_contract = _target_variable_contract(
        target,
        build_id=build_id,
        block_signature=str(target.attrs.get("block_signature_sha256", "")),
    )
    contract_path = (
        TARGET_VARIABLE_CONTRACT
        if not noncanonical and not args.centroid_mask_quick
        else TARGET_VARIABLE_CONTRACT.with_name(
            TARGET_VARIABLE_CONTRACT.stem
            + f"_{noncanonical_suffix or 'centroid_quick'}"
            + TARGET_VARIABLE_CONTRACT.suffix
        )
    )
    written_contract = write_csv_atomic(variable_contract, contract_path)
    promoted_manifest = json.loads(block_manifest.read_text(encoding="utf-8"))
    promoted_manifest.update(
        {
            "blocks_complete": True,
            "build_complete": True,
            "promotion_status": "promoted_after_deep_validation",
            "promoted_utc": datetime.now(timezone.utc).isoformat(),
            "promoted_target": str(destination.relative_to(ROOT)).replace("\\", "/"),
            "promoted_pixel_inventory": str(written_pixels.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "promoted_target_data_state_sha256": _zarr_state_fingerprint(destination),
            "promoted_target_data_content_sha256": _zarr_content_fingerprint(
                destination
            ),
            "promoted_pixel_inventory_sha256": sha256_file(written_pixels),
            "build_id": build_id,
            "target_contract_version": str(
                target.attrs.get("target_contract_version", "")
            ),
            "promoted_target_grid_hash_sha256": str(
                target.attrs.get("grid_hash_sha256", "")
            ),
            "target_variable_contract": str(
                written_contract.relative_to(ROOT)
            ).replace("\\", "/"),
            "target_variable_contract_sha256": sha256_file(written_contract),
            "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            "target_module_sha256": sha256_file(
                ROOT / "src/nino_brasil/targets/chirps_native.py"
            ),
            "deep_validation": staged_validation.as_dict(),
            "pixel_mask_sha256": _pixel_mask_fingerprint(pixels),
        }
    )
    _atomic_json(block_manifest, promoted_manifest)
    print(f"[target] {destination.relative_to(ROOT)}")
    print(f"[pixels] {written_pixels.relative_to(ROOT)}")
    print(f"[numeric contract] {written_contract.relative_to(ROOT)}")
    print(f"[restart manifest] {block_manifest.relative_to(ROOT)}")
    return 0


def validate_promoted_target(path: Path = OUTPUT) -> dict[str, object]:
    """Validate target bytes, inventory, contract and promotion lineage."""

    problems: list[str] = []
    if not path.is_dir():
        return {"valid": False, "problems": [f"missing_target:{path}"]}
    dataset = xr.open_zarr(path, consolidated=None)
    try:
        target_validation = validate_canonical_target(dataset)
        attrs = dict(dataset.attrs)
        variables = set(dataset.data_vars)
        sizes = dict(dataset.sizes)
    finally:
        dataset.close()
    if not target_validation.valid:
        problems.extend(target_validation.errors)
    signature = str(attrs.get("block_signature_sha256", "")).strip()
    build_id = str(attrs.get("build_id", "")).strip()
    if not signature or len(signature) != 64:
        problems.append("target_block_signature_missing_or_invalid")
    if not build_id:
        problems.append("target_build_id_missing")
    if attrs.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        problems.append("target_contract_version_mismatch")
    if attrs.get("deep_validation_passed") is not True:
        problems.append("target_deep_validation_stamp_missing")
    manifest_path = WEEKLY_BLOCK_ROOT / signature[:16] / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "target_validation": target_validation.as_dict(),
            "problems": [*problems, f"invalid_build_manifest:{exc}"],
            "manifest": str(manifest_path),
        }

    def recorded_path(key: str) -> Path | None:
        value = str(manifest.get(key) or "").strip()
        if not value:
            problems.append(f"manifest_missing_path:{key}")
            return None
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            problems.append(f"manifest_path_outside_project:{key}")
            return None
        return resolved

    if manifest.get("build_complete") is not True:
        problems.append("manifest_build_incomplete")
    if manifest.get("promotion_status") != "promoted_after_deep_validation":
        problems.append("manifest_promotion_status_invalid")
    expected_scalars = {
        "signature_sha256": signature,
        "build_id": build_id,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "promoted_target_grid_hash_sha256": str(
            attrs.get("grid_hash_sha256", "")
        ),
    }
    for key, expected in expected_scalars.items():
        if str(manifest.get(key) or "") != expected:
            problems.append(f"manifest_value_mismatch:{key}")
    promoted_target = recorded_path("promoted_target")
    if promoted_target != path.resolve():
        problems.append("manifest_promoted_target_mismatch")
    pixel_path = recorded_path("promoted_pixel_inventory")
    contract_path = recorded_path("target_variable_contract")

    current_state_sha = _zarr_state_fingerprint(path)
    if manifest.get("promoted_target_data_state_sha256") != current_state_sha:
        problems.append("promoted_target_state_fingerprint_mismatch")
    current_content_sha = _zarr_content_fingerprint(path)
    if manifest.get("promoted_target_data_content_sha256") != current_content_sha:
        problems.append("promoted_target_content_fingerprint_mismatch")
    if manifest.get("builder_script_sha256") != sha256_file(Path(__file__).resolve()):
        problems.append("builder_script_fingerprint_mismatch")
    target_module = ROOT / "src/nino_brasil/targets/chirps_native.py"
    if manifest.get("target_module_sha256") != sha256_file(target_module):
        problems.append("target_module_fingerprint_mismatch")

    if pixel_path is None or not pixel_path.is_file():
        problems.append("promoted_pixel_inventory_missing")
    else:
        if manifest.get("promoted_pixel_inventory_sha256") != sha256_file(pixel_path):
            problems.append("promoted_pixel_inventory_hash_mismatch")
        try:
            pixels = pd.read_csv(pixel_path)
            required = {
                "pixel_id",
                "grid_row",
                "grid_column",
                "lat",
                "lon",
                "brazil_fraction",
                "brazil_center",
                "grid_hash",
            }
            if required.difference(pixels.columns):
                problems.append("promoted_pixel_inventory_schema_mismatch")
            elif len(pixels) != int(sizes.get("latitude", 0)) * int(
                sizes.get("longitude", 0)
            ):
                problems.append("promoted_pixel_inventory_row_count_mismatch")
            elif pixels["pixel_id"].duplicated().any():
                problems.append("promoted_pixel_inventory_duplicate_pixel_id")
            else:
                hashes = set(pixels["grid_hash"].dropna().astype(str))
                if hashes != {str(attrs.get("grid_hash_sha256", ""))}:
                    problems.append("promoted_pixel_inventory_grid_hash_mismatch")
                if manifest.get("pixel_mask_sha256") != _pixel_mask_fingerprint(pixels):
                    problems.append("promoted_pixel_mask_fingerprint_mismatch")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"promoted_pixel_inventory_unreadable:{type(exc).__name__}")

    if contract_path is None or not contract_path.is_file():
        problems.append("target_variable_contract_missing")
    else:
        if manifest.get("target_variable_contract_sha256") != sha256_file(contract_path):
            problems.append("target_variable_contract_hash_mismatch")
        try:
            contract = pd.read_csv(contract_path)
            required = {
                "build_id",
                "variable",
                "grid_hash_sha256",
                "target_contract_version",
                "block_signature_sha256",
            }
            if required.difference(contract.columns):
                problems.append("target_variable_contract_schema_mismatch")
            elif set(contract["variable"].astype(str)) != variables:
                problems.append("target_variable_contract_variable_set_mismatch")
            else:
                identities = {
                    "build_id": build_id,
                    "grid_hash_sha256": str(attrs.get("grid_hash_sha256", "")),
                    "target_contract_version": TARGET_CONTRACT_VERSION,
                    "block_signature_sha256": signature,
                }
                for column, expected in identities.items():
                    if set(contract[column].dropna().astype(str)) != {expected}:
                        problems.append(f"target_variable_contract_identity:{column}")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"target_variable_contract_unreadable:{type(exc).__name__}")

    return {
        "valid": not problems,
        "target_validation": target_validation.as_dict(),
        "problems": problems,
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "build_id": build_id,
        "block_signature_sha256": signature,
        "target_data_content_sha256": current_content_sha,
        "pixel_inventory": (
            str(pixel_path.relative_to(ROOT)).replace("\\", "/")
            if pixel_path is not None and pixel_path.exists()
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--skip-spi", action="store_true", help="skip provisional gamma SPI layers")
    parser.add_argument("--skip-extremes", action="store_true", help="skip weekly extreme-index layers")
    parser.add_argument(
        "--latitude-block-size",
        type=int,
        default=4,
        help="native latitude rows per restartable weekly block (default: 4; RAM-safe on 32 GiB)",
    )
    parser.add_argument(
        "--centroid-mask-quick",
        action="store_true",
        help="write a separately named quick artifact; never the canonical target",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="archive an existing output before promoting the validated staged build",
    )
    parser.add_argument(
        "--rebuild-daily",
        action="store_true",
        help="archive and rebuild the persistent pre-resample native daily cube",
    )
    parser.add_argument(
        "--resume-daily-staging",
        type=Path,
        help="resume an explicitly named validated .staging-* daily Zarr after interruption",
    )
    parser.add_argument(
        "--resume-through-date",
        help="last verified complete YYYY-MM-DD when resuming an older staging without checkpoint attrs",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit-legacy", action="store_true")
    parser.add_argument("--quarantine-invalid-legacy", action="store_true")
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="disable the default one-latitude-block-per-process memory isolation",
    )
    parser.add_argument("--block-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-new-weekly-blocks", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--only-weekly-block-start", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--parallel-block-workers",
        type=int,
        default=1,
        help="reserved; current audited default is one isolated worker",
    )
    args = parser.parse_args(raw_argv)
    warnings.filterwarnings(
        "ignore", message="All-NaN slice encountered", category=RuntimeWarning
    )
    warnings.filterwarnings(
        "ignore", message="invalid value encountered in divide", category=RuntimeWarning
    )
    if args.resume_through_date and not args.resume_daily_staging:
        parser.error("--resume-through-date requires --resume-daily-staging")

    if args.audit_legacy or args.quarantine_invalid_legacy:
        report = audit_legacy_parquet()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.quarantine_invalid_legacy:
            if report.get("valid") is not False:
                raise ValueError("Legacy cache was not proven invalid; refusing quarantine.")
            for path in (LEGACY_Z, LEGACY_PIXELS):
                if path.exists():
                    archived = _archive_existing(
                        path,
                        reason="legacy Phase 4 cache failed robust-z validation",
                    )
                    print(f"[archive] {archived.relative_to(ROOT)}")
        if not args.validate_only:
            return 0

    if args.validate_only:
        report = validate_promoted_target(OUTPUT)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["valid"] else 2

    if args.parallel_block_workers < 1:
        parser.error("--parallel-block-workers must be positive")
    if args.parallel_block_workers > 1:
        parser.error(
            "parallel CHIRPS workers are disabled: an 8-row block can exceed 16 GiB; "
            "use the audited one-block-per-process default"
        )

    if not args.single_process and not args.block_worker:
        worker_argv = list(raw_argv)
        daily_mutation_requested = bool(
            args.rebuild_daily or args.resume_daily_staging is not None
        )
        if daily_mutation_requested:
            bootstrap = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *worker_argv,
                    "--block-worker",
                    "--max-new-weekly-blocks",
                    "1",
                ],
                cwd=ROOT,
                check=False,
            )
            if bootstrap.returncode not in {0, PARTIAL_BLOCK_EXIT_CODE}:
                return int(bootstrap.returncode)
            if bootstrap.returncode == 0:
                return 0
            worker_argv = _drop_cli_options(
                worker_argv,
                flags={"--rebuild-daily"},
                valued={"--resume-daily-staging", "--resume-through-date"},
            )
        child_args = [
            sys.executable,
            str(Path(__file__).resolve()),
            *worker_argv,
            "--block-worker",
            "--max-new-weekly-blocks",
            "1",
        ]
        print(
            "[CHIRPS] isolated workers enabled: one new latitude block per process",
            flush=True,
        )
        while True:
            completed = subprocess.run(child_args, cwd=ROOT, check=False)
            if completed.returncode == PARTIAL_BLOCK_EXIT_CODE:
                continue
            return int(completed.returncode)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
