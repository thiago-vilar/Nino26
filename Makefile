PYTHON := .venv-wsl/bin/python

.PHONY: fase1 fase2 fase3 fase4 fase5 fase6 fase7 fase8

fase1:
	$(PYTHON) scripts/run_full_download_pipeline.py --execute --stop-on-error

fase2:
	$(PYTHON) scripts/run_fase2_all.py

fase3:
	$(PYTHON) scripts/run_fase3_nino_all.py

fase4:
	$(PYTHON) scripts/run_fase4_all.py

fase5:
	$(PYTHON) scripts/run_fase5_all.py

fase6:
	$(PYTHON) scripts/run_fase6_all.py

fase7:
	$(PYTHON) scripts/run_fase7_all.py

fase8:
	$(PYTHON) scripts/run_fase8_all.py
