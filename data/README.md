# Data directory

This directory is organized by processing stage.

- `raw/`: downloaded files exactly as obtained from the source.
- `interim/`: standardized files after unit, calendar and domain corrections.
- `processed/`: analysis-ready Zarr, Parquet and GeoTIFF outputs.
- `catalog/`: dataset metadata and provenance.

Large data files should not be committed to Git.
