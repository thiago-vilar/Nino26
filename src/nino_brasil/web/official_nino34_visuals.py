from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OfficialNino34Visual:
    name: str
    url: str
    source_page: str
    output_name: str
    description: str


@dataclass(frozen=True)
class OfficialNino34VisualSyncOutput:
    output_dir: Path
    manifest_path: Path
    files: list[Path]
    rows: int


DEFAULT_OFFICIAL_NINO34_VISUALS = [
    OfficialNino34Visual(
        name="noaa_psl_nino34_timeseries",
        url="https://psl.noaa.gov/enso/dashboard/img/nino34.png",
        source_page="https://psl.noaa.gov/enso/dashboard.html",
        output_name="noaa_psl_nino34_timeseries.png",
        description="NOAA/PSL official Nino 3.4 SST anomaly dashboard time-series graphic.",
    ),
    OfficialNino34Visual(
        name="noaa_psl_nino34_event_panel",
        url="https://psl.noaa.gov/enso/dashboard/img/nino34longpanel.png",
        source_page="https://psl.noaa.gov/enso/dashboard.html",
        output_name="noaa_psl_nino34_event_panel.png",
        description="NOAA/PSL official Nino 3.4 current event versus historical event panel.",
    ),
]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_official_nino34_visuals(
    output_dir: Path,
    *,
    timeout_seconds: int = 60,
    dry_run: bool = False,
) -> OfficialNino34VisualSyncOutput:
    """Mirror official Nino 3.4 graphics for visual comparison only.

    These files are intentionally kept under ``docs/assets`` rather than
    ``data/processed`` because they are not pipeline metrics, labels, event
    definitions, or model inputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "official_nino34_visuals_manifest.json"
    files: list[Path] = []
    rows: list[dict[str, object]] = []

    for visual in DEFAULT_OFFICIAL_NINO34_VISUALS:
        output_path = output_dir / visual.output_name
        files.append(output_path)
        base_row: dict[str, object] = {
            **asdict(visual),
            "output_path": str(output_path),
            "source_contract": "official NOAA/PSL graphic mirrored for visual comparison only",
            "pipeline_role": "visual_reference_only_not_metric_not_label_not_model_input",
            "metric_source": False,
            "fetched_at_utc": None,
            "size_bytes": None,
            "sha256": None,
        }
        if dry_run:
            base_row["exists"] = output_path.exists()
            rows.append(base_row)
            continue

        request = Request(
            visual.url,
            headers={"User-Agent": "NINO-BRASIL visual-reference-sync/1.0"},
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")

        if len(content) < 1024:
            raise ValueError(f"Downloaded official Nino 3.4 visual is unexpectedly small: {visual.url}")
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Downloaded official Nino 3.4 visual is not a PNG: {visual.url} ({content_type})")

        output_path.write_bytes(content)
        base_row.update(
            {
                "exists": True,
                "fetched_at_utc": _utc_now(),
                "size_bytes": len(content),
                "sha256": _sha256_bytes(content),
                "content_type": content_type,
            }
        )
        rows.append(base_row)

    manifest = {
        "created_at_utc": _utc_now(),
        "source_contract": "official graphics for visual comparison only; local OISST remains the metric source",
        "pipeline_role": "excluded_from_phase3_metrics_events_p90_and_diagnostics",
        "visuals": rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return OfficialNino34VisualSyncOutput(
        output_dir=output_dir,
        manifest_path=manifest_path,
        files=files,
        rows=len(rows),
    )
