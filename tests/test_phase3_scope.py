from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
