from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_ocean_monthly import ORAS5_VARIABLES, oras5_feature_path, oras5_variable_zarr_path
from nino_brasil.data.download_ocean_daily import _validate_canonical_ocean_grid


DAILY_ROOT = ROOT / "data/processed/zarr/ocean_daily"
DAILY_FEATURE_ROOT = ROOT / "data/processed/zarr/features/ocean_daily"
MONTHLY_ROOT = ROOT / "data/processed/zarr/ocean_monthly/oras5"
MONTHLY_FEATURE_ROOT = ROOT / "data/processed/zarr/features/ocean_monthly/oras5"
REPORT_PATH = ROOT / "data/audit/ocean_phase2_audit.json"
TRANSITION_PATH = ROOT / "data/processed/parquet/ocean_source_transition_audit.csv"

DAILY_REQUIRED = {"potential_temperature", "salinity", "sea_surface_height"}
FEATURE_REQUIRED = {
    "d20_nino34_mean_m",
    "ohc_0_300_nino34_j_m2",
    "ohc_0_700_nino34_j_m2",
    "wwv_equatorial_pacific_m3",
    "thermocline_tilt_m",
    "thermocline_tilt_slope_m_per_degree",
    "sss_nino34_mean",
    "ssh_nino34_mean_m",
    "ocean_source_code",
}


def _check_daily_store(path: Path, expected: pd.DatetimeIndex) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing:{path}"]
    with xr.open_zarr(path, consolidated=None) as ds:
        missing = DAILY_REQUIRED.difference(ds.data_vars)
        if missing:
            errors.append(f"variables:{path}:{sorted(missing)}")
        actual = pd.DatetimeIndex(ds["time"].values).normalize()
        if not actual.equals(expected):
            errors.append(f"calendar:{path}:expected={len(expected)}:actual={len(actual)}")
        if str(ds.attrs.get("temporal_transform")) != "none":
            errors.append(f"temporal_transform:{path}:{ds.attrs.get('temporal_transform')}")
        try:
            _validate_canonical_ocean_grid(ds)
        except (KeyError, ValueError) as exc:
            errors.append(f"canonical_grid:{path}:{exc}")
    return errors


def _check_feature_store(path: Path, expected: pd.DatetimeIndex) -> list[str]:
    if not path.exists():
        return [f"missing:{path}"]
    errors: list[str] = []
    with xr.open_zarr(path, consolidated=None) as ds:
        missing = FEATURE_REQUIRED.difference(ds.data_vars)
        if missing:
            errors.append(f"feature_variables:{path}:{sorted(missing)}")
        actual = pd.DatetimeIndex(ds["time"].values).normalize()
        if not actual.equals(expected):
            errors.append(f"feature_calendar:{path}:expected={len(expected)}:actual={len(actual)}")
    return errors


def _daily_paths(source: str, year: int) -> tuple[Path, Path]:
    if source == "noaa_ufs":
        cube = DAILY_ROOT / source / str(year) / f"noaa_ufs_equatorial_pacific_{year}_daily.zarr"
    else:
        cube = DAILY_ROOT / source / str(year) / f"glorys12_equatorial_pacific_{year}_daily_0p25.zarr"
    feature = DAILY_FEATURE_ROOT / source / str(year) / f"{source}_ocean_features_{year}_daily.zarr"
    return cube, feature


def audit_core(args: argparse.Namespace) -> dict[str, object]:
    errors: list[str] = []
    for year in range(1981, 1993):
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        cube, feature = _daily_paths("noaa_ufs", year)
        errors.extend(_check_daily_store(cube, expected))
        errors.extend(_check_feature_store(feature, expected))

    my_end = pd.Timestamp(args.glorys_my_end).normalize()
    for year in range(1993, my_end.year + 1):
        end = min(pd.Timestamp(f"{year}-12-31"), my_end)
        expected = pd.date_range(f"{year}-01-01", end, freq="D")
        cube, feature = _daily_paths("glorys12", year)
        errors.extend(_check_daily_store(cube, expected))
        errors.extend(_check_feature_store(feature, expected))

    operational_end = pd.Timestamp(args.operational_end).normalize()
    operational_start = my_end + pd.Timedelta(days=1)
    if operational_end >= operational_start:
        slug = f"{operational_start:%Y%m%d}_{operational_end:%Y%m%d}"
        cube = DAILY_ROOT / "glorys12_operational" / str(operational_start.year) / f"glorys12_operational_{slug}_daily_0p25.zarr"
        feature = DAILY_FEATURE_ROOT / "glorys12_operational" / str(operational_start.year) / f"glorys12_operational_ocean_features_{slug}_daily.zarr"
        expected = pd.date_range(operational_start, operational_end, freq="D")
        errors.extend(_check_daily_store(cube, expected))
        errors.extend(_check_feature_store(feature, expected))

    oras_end = pd.Timestamp(args.oras_end).to_period("M").to_timestamp()
    for year in range(1981, oras_end.year + 1):
        last_month = oras_end.month if year == oras_end.year else 12
        expected = pd.DatetimeIndex([pd.Timestamp(year=year, month=month, day=1) for month in range(1, last_month + 1)])
        for variable in ORAS5_VARIABLES:
            path = oras5_variable_zarr_path(MONTHLY_ROOT, year, variable)
            if not path.exists():
                errors.append(f"missing:{path}")
                continue
            with xr.open_zarr(path, consolidated=None) as ds:
                actual = pd.DatetimeIndex(ds["time"].values)
                if not actual.equals(expected):
                    errors.append(f"monthly_calendar:{path}:expected={len(expected)}:actual={len(actual)}")
                if str(ds.attrs.get("source_frequency")) != "monthly_mean":
                    errors.append(f"monthly_frequency:{path}:{ds.attrs.get('source_frequency')}")
                try:
                    _validate_canonical_ocean_grid(ds)
                except (KeyError, ValueError) as exc:
                    errors.append(f"monthly_canonical_grid:{path}:{exc}")
        feature = oras5_feature_path(MONTHLY_FEATURE_ROOT, year)
        if not feature.exists():
            errors.append(f"missing:{feature}")

    result = {
        "status": "complete" if not errors else "incomplete",
        "daily_period": f"1981-01-01/{operational_end:%Y-%m-%d}",
        "glorys_multiyear_end": f"{my_end:%Y-%m-%d}",
        "oras_monthly_end": f"{oras_end:%Y-%m}",
        "error_count": len(errors),
        "errors": errors,
        "scientific_contract": {
            "monthly_promoted_to_daily": False,
            "forecast_days_in_historical_series": False,
            "daily_sources": ["NOAA_UFS", "GLORYS12_MY", "GLO12_ANFC_analysis_only"],
            "monthly_source": "ORAS5",
            "canonical_horizontal_grid_degrees": 0.25,
            "canonical_grid_required_for": ["NOAA_UFS", "GLORYS12_MY", "GLO12_ANFC", "ORAS5"],
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ocean Phase 2 audit: {result['status']}; errors={len(errors)}; report={REPORT_PATH}")
    for error in errors[:30]:
        print(error)
    return result


def audit_transition(years: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in years:
        ufs_path = DAILY_FEATURE_ROOT / "noaa_ufs" / str(year) / f"noaa_ufs_ocean_features_{year}_daily.zarr"
        glorys_path = DAILY_FEATURE_ROOT / "glorys12" / str(year) / f"glorys12_ocean_features_{year}_daily.zarr"
        if not ufs_path.exists() or not glorys_path.exists():
            rows.append({"year": year, "status": "missing_overlap_store"})
            continue
        with xr.open_zarr(ufs_path, consolidated=None) as ufs, xr.open_zarr(glorys_path, consolidated=None) as glorys:
            common = sorted(set(ufs.data_vars).intersection(glorys.data_vars).difference({"ocean_source_code"}))
            left, right = xr.align(ufs[common], glorys[common], join="inner")
            for variable in common:
                difference = left[variable] - right[variable]
                rows.append(
                    {
                        "year": year,
                        "status": "ok",
                        "variable": variable,
                        "matched_days": int(difference.sizes.get("time", 0)),
                        "ufs_minus_glorys_mean": float(difference.mean(skipna=True)),
                        "rmse": float(np.sqrt((difference**2).mean(skipna=True))),
                    }
                )
    frame = pd.DataFrame(rows)
    TRANSITION_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TRANSITION_PATH, index=False)
    print(f"transition audit: {TRANSITION_PATH}; rows={len(frame)}")
    return frame


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Audit the dual-frequency ocean contract required to close Phase 2.")
    root.add_argument("--glorys-my-end", default="2026-05-26")
    root.add_argument("--operational-end", default=(pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    root.add_argument("--oras-end", default="2026-05-01")
    root.add_argument("--overlap-year", type=int, action="append", default=[])
    return root


def main() -> int:
    args = parser().parse_args()
    result = audit_core(args)
    if args.overlap_year:
        transition = audit_transition(args.overlap_year)
        transition_ok = not transition.empty and bool((transition["status"] == "ok").all())
        if not transition_ok:
            errors = list(result.get("errors", []))
            errors.append("source_transition_audit:missing_or_invalid_overlap_stores")
            result["errors"] = errors
            result["error_count"] = len(errors)
            result["status"] = "incomplete"
            REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
