# Wildfire Data Processing Script

This Python script processes wildfire data by filtering and clustering based on various parameters. It is designed to handle CSV files containing wildfire data, allowing users to specify date ranges, geographic coordinates, and other criteria for data selection and analysis. Images from SentinelHub are downloaded based on the filtered coordinates and date ranges.
## Requirements

- Python 3.x
- Libraries:
  - `geopandas`
  - `pandas`
  - `numpy`
  - `sklearn`
  - `shapely`
  - `logging`
  - `argparse`

## Usage

To use this script, you need to provide specific command-line arguments for data processing. The arguments include directories for input CSV files, date range for filtering, geographic coordinates for spatial filtering, and other parameters.

**Command-Line Arguments:**

- `--csv_in_dir`: Directory containing CSV files.
- `--start_date`: Start date for filtering (format: YYYY-MM-DD).
- `--end_date`: End date for filtering (format: YYYY-MM-DD).
- `--longitude`: Longitude for geographic filtering (optional).
- `--latitude`: Latitude for geographic filtering (optional).
- `--max_rows`: Maximum rows to include in the final DataFrame.
- `--backward_days`: Days to go back from the fire date.
- `--forward_days`: Days to go forward from the fire date.
- `--fire_confidence`: List of fire confidence levels to filter (e.g., 'h', 'n').
- `--filter_buffer`: Buffer distance for geographic filtering.

**Example Command:**

```bash
python script_name.py --csv_in_dir "path/to/csv" --start_date "2021-01-01" --end_date "2021-12-31" --max_rows 1000 --backward_days 7 --forward_days 7
```

## Function prepare_data
The prepare_data function is the core of this script. It prepares and filters wildfire data based on the provided parameters. The function reads VIIRS/MODIS CSV files from the specified directory, filters data based on date range, geographic coordinates, and other criteria, and then processes the data for further analysis.

