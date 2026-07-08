from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "notebooks" / "fase3" / "fase3_utils.py"


def _load_phase3_utils():
    spec = importlib.util.spec_from_file_location("phase3_utils_contract", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Phase3ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.u = _load_phase3_utils()

    def test_phase3_public_labels_exclude_salinity(self) -> None:
        public_text = " ".join(
            [
                " ".join(self.u.VAR_LABELS),
                " ".join(self.u.VAR_LABELS.values()),
                " ".join(self.u.VAR_SHORT),
                " ".join(self.u.VAR_SHORT.values()),
                self.u.sources_note().to_string(index=False),
            ]
        )
        self.assertNotIn("sss", public_text.lower())

    def test_phase3_wind_contract_uses_anomaly_not_raw_proxy(self) -> None:
        self.assertIn("tau_x_anom_nino34_pa", self.u.VAR_LABELS)
        self.assertNotIn("tau_x_proxy_nino34_pa", self.u.VAR_LABELS)
        self.assertIn("anomalia diaria", self.u.sources_note().to_string(index=False).lower())

    def test_phase3_official_longitude_references_are_we_not_360(self) -> None:
        references = " ".join(self.u.CAIXAS.values())
        self.assertIn("5N-5S, 170W-120W", references)
        self.assertIn("5N-5S, 160E-150W", references)
        self.assertIn("120E-80W", references)
        self.assertNotIn("0-360", references)

    def test_phase3_longitude_axis_runs_west_to_east(self) -> None:
        fig, ax = plt.subplots()
        try:
            self.u.format_lon_axis(ax)
            self.assertEqual(tuple(ax.get_xlim()), (120.0, 280.0))
            self.assertEqual(list(ax.get_xticks()), [120, 160, 200, 240, 280])
            self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["120E", "160E", "160W", "120W", "80W"])
        finally:
            plt.close(fig)

    def test_phase3_elnino_mean_groups_are_fixed_and_ordered(self) -> None:
        self.assertEqual(
            self.u.ELNINO_MEAN_GROUP_ORDER,
            ("forte_p90", "super_p95", "eventos_gt_p90"),
        )
        table = self.u.elnino_mean_group_table()
        self.assertEqual(table["grupo"].tolist(), list(self.u.ELNINO_MEAN_GROUP_ORDER))
        self.assertIn(">P90 e <P95", table.loc[table["grupo"] == "forte_p90", "definicao"].iloc[0])
        self.assertIn(">P95", table.loc[table["grupo"] == "super_p95", "definicao"].iloc[0])

    def test_phase3_elnino_mean_groups_include_classes_and_gt90_mean(self) -> None:
        events = pd.DataFrame(
            {
                "event_id": ["a", "b", "c"],
                "classe_p90_p95": ["forte_p90", "super_p95", "forte_p90"],
            }
        )
        groups = self.u.elnino_mean_groups(events)
        self.assertEqual(list(groups), list(self.u.ELNINO_MEAN_GROUP_ORDER))
        self.assertEqual(groups["forte_p90"]["event_id"].tolist(), ["a", "c"])
        self.assertEqual(groups["super_p95"]["event_id"].tolist(), ["b"])
        self.assertEqual(groups["eventos_gt_p90"]["event_id"].tolist(), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
