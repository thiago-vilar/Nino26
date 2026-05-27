# Data directory

This directory is organized by processing stage.

- `raw/`: downloaded files exactly as obtained from the source.
- `interim/`: standardized files after unit, calendar and domain corrections.
- `processed/`: analysis-ready Zarr, Parquet and GeoTIFF outputs.
- `catalog/`: dataset metadata and provenance.

Large data files should not be committed to Git.

Main raw data areas:

- `raw/oras/`: ORAS/ORAS5 ocean reanalysis.
- `raw/ctd_noaa/`: NOAA/WOD CTD profiles.
- `raw/era5/`: ERA5 atmospheric reanalysis.
- `raw/cpc_noaa/`: CPC/NOAA precipitation, SST and OLR products.
- `raw/ibge/`: official Brazil boundaries and territorial masks.
