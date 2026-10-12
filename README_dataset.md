# Quick Cloud Dataset Creation for DeltaSAT

This directory provides tools to quickly create cloud-ready datasets for DeltaSAT analysis.

## Quick Start

### 1. Install Dependencies

```bash
pip install earthengine-api geemap fastapi uvicorn
```

### 2. Authenticate with Google Earth Engine

```bash
# Run once to authenticate
earthengine authenticate

# Or use a service account (recommended for production)
export EE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
export EE_KEY_FILE=/path/to/key.json
export EE_PROJECT=your-gcp-project
```

### 3. Create a Dataset

#### Interactive Mode (Easiest)

```bash
cd demo
python make_cloud_dataset.py
# Follow the prompts
```

#### Command-Line Mode

```bash
# Basic dataset with defaults
python make_cloud_dataset.py \
  --bbox 90.0,22.0,91.0,23.0 \
  --years 2020,2022 \
  --output ./my_dataset

# Sentinel-2 only (faster, 10m resolution)
python make_cloud_dataset.py \
  --bbox 90.0,22.0,91.0,23.0 \
  --years 2020,2022 \
  --sensors s2 \
  --output ./s2_dataset

# Just generate preview tile layers (no export)
python make_cloud_dataset.py \
  --bbox 90.0,22.0,91.0,23.0 \
  --years 2020,2022 \
  --preview-only
```

### 4. Use with DeltaSAT Backend

Once you have a dataset, you can use it with the FastAPI backend:

```bash
# Start the backend (MOCK mode for testing)
MOCK_MODE=1 uvicorn app:app --reload --port 8000

# Or with real processing
MOCK_MODE=0 MODULES_DIR="/path/to/codes/Modules" uvicorn app:app --port 8000
```

## Dataset Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bbox` | Bounding box as "w,s,e,n" or "w s e n" | Required |
| `--years` | Year range as "2020,2022" or "2020-2022" | Required |
| `--sensors` | Sensor type: landsat, s2, or both | both |
| `--output` | Output directory | ./cloud_dataset |
| `--max-cloud` | Max cloud cover percentage | 20 |
| `--scale` | Export scale in meters | 10 |
| `--preview-only` | Generate preview layers only | False |
| `--interactive` | Use interactive prompts | False |

## Output Structure

When you run a full export, the script creates:

```
output_dir/
├── metadata.json          # Dataset metadata (bbox, years, sensors, etc.)
├── sentinel2/            # Sentinel-2 MNDWI GeoTIFFs (if --sensors s2 or both)
│   └── mndwi_YYYY-MM-DD.tif
└── landsat/              # Landsat data (requires full pipeline)
    └── README.txt
```

## Preview Layers

The `--preview-only` option generates tile layer URLs for Leaflet without exporting data. Output includes:

- `preview_layers.json` - JSON with all layer configurations
- Console output with direct URLs for each layer

Layers generated:
1. **MNDWI (median)** - Modified Normalized Difference Water Index composite
2. **Water mask (MNDWI>0)** - Binary water detection
3. **JRC water occurrence** - JRC Global Surface Water occurrence layer

## Notes

- **Sentinel-2**: Fully supported, 10m resolution, faster processing
- **Landsat**: Requires the full DeltaSAT pipeline modules (`MODULES_DIR`)
- **Authentication**: You must authenticate with GEE before running
- **Cloud filtering**: Use `--max-cloud` to control cloud cover threshold

## Example: Ganges Delta

```bash
# Create dataset for Ganges delta region
python make_cloud_dataset.py \
  --bbox 89.0,21.0,91.0,23.5 \
  --years 2015,2020 \
  --sensors s2 \
  --max-cloud 15 \
  --output ./ganges_dataset
```

## Example: Preview for Web Map

```bash
# Generate preview layers for immediate use in Leaflet
python make_cloud_dataset.py \
  --bbox 90.0,22.0,91.0,23.0 \
  --years 2020,2022 \
  --preview-only

# The output preview_layers.json can be loaded directly into your frontend
```
