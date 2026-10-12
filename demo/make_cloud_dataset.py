#!/usr/bin/env python3
"""
make_cloud_dataset.py - Quick script to create cloud-ready datasets for DeltaSAT.

This provides a simple CLI to generate the datasets needed for DeltaSAT analysis:
- MNDWI composites from Sentinel-2 or Landsat
- Export to GeoTIFFs in the correct format for the pipeline
- Preview tile layers for the web map

Usage:
    # Basic usage with interactive prompts
    python make_cloud_dataset.py

    # Or with command-line arguments
    python make_cloud_dataset.py --bbox 90.0,22.0,91.0,23.0 --years 2020,2022 --output ./my_dataset

    # For Sentinel-2 only (faster, 10m resolution)
    python make_cloud_dataset.py --bbox 90.0,22.0,91.0,23.0 --years 2020,2022 --sensors s2 --output ./s2_dataset

    # For preview tiles (no export)
    python make_cloud_dataset.py --bbox 90.0,22.0,91.0,23.0 --years 2020,2022 --preview-only

Requirements:
    pip install earthengine-api geemap
    earthengine authenticate  # Run once to authenticate
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_bbox(bbox_str):
    """Parse bbox string like 'w,s,e,n' or 'w s e n' into dict."""
    parts = bbox_str.replace(',', ' ').split()
    if len(parts) != 4:
        raise ValueError(f"Bounding box must have 4 values: w s e n, got: {bbox_str}")
    w, s, e, n = map(float, parts)
    return {"w": w, "s": s, "e": e, "n": n}


def parse_years(years_str):
    """Parse years string like '2020,2022' or '2020-2022' into tuple."""
    years_str = years_str.replace('-', ',')
    parts = years_str.split(',')
    if len(parts) != 2:
        raise ValueError(f"Years must be 2 values: start,end, got: {years_str}")
    return tuple(int(p) for p in parts)


def get_user_input():
    """Interactive input collection."""
    print("=" * 60)
    print("DeltaSAT Cloud Dataset Generator")
    print("=" * 60)
    
    # Bounding box
    bbox_str = input("Enter bounding box (w,s,e,n or w s e n): ").strip()
    bbox = parse_bbox(bbox_str)
    print(f"  BBox: {bbox}")
    
    # Years
    years_str = input("Enter year range (2020,2022 or 2020-2022): ").strip()
    years = parse_years(years_str)
    print(f"  Years: {years[0]} to {years[1]}")
    
    # Sensors
    sensors_input = input("Sensors [landsat, s2, both] (default: both): ").strip().lower()
    if sensors_input in ['', 'both', 'all']:
        sensors = ['landsat', 's2']
    elif sensors_input in ['s2', 'sentinel', 'sentinel2']:
        sensors = ['s2']
    elif sensors_input in ['landsat', 'l']:
        sensors = ['landsat']
    else:
        sensors = ['landsat', 's2']
    print(f"  Sensors: {sensors}")
    
    # Output directory
    output_dir = input("Output directory (default: ./cloud_dataset): ").strip()
    if not output_dir:
        output_dir = "./cloud_dataset"
    print(f"  Output: {output_dir}")
    
    # Max cloud cover
    cloud_input = input("Max cloud cover % (default: 20): ").strip()
    max_cloud = int(cloud_input) if cloud_input else 20
    print(f"  Max cloud: {max_cloud}%")
    
    # Preview only
    preview_only = input("Preview only? (y/n, default: n): ").strip().lower()
    preview_only = preview_only in ['y', 'yes']
    print(f"  Preview only: {preview_only}")
    
    return bbox, years, sensors, output_dir, max_cloud, preview_only


def init_ee():
    """Initialize Earth Engine."""
    import ee
    
    project = os.getenv("EE_PROJECT")
    sa = os.getenv("EE_SERVICE_ACCOUNT")
    key = os.getenv("EE_KEY_FILE")
    
    if sa and key:
        creds = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(creds, project=project)
    else:
        try:
            ee.Initialize(project=project)
        except Exception as e:
            print(f"\n{'='*60}")
            print("GEE Authentication Required")
            print("="*60)
            print(f"Error: {e}")
            print("\nTo authenticate:")
            print("  1. Run: earthengine authenticate")
            print("  2. Or set EE_SERVICE_ACCOUNT and EE_KEY_FILE env vars")
            print("  3. Optional: Set EE_PROJECT for your GCP project")
            print("\nThen re-run this script.")
            sys.exit(1)
    
    print("\n✓ GEE initialized")


def generate_preview(bbox, years, sensors, max_cloud=20):
    """Generate preview tile layers."""
    import gee
    
    print("\nGenerating preview layers...")
    layers = gee.preview_layers(bbox, years[0], years[1], 
                                sensors=sensors, max_cloud=max_cloud)
    
    print("\n" + "="*60)
    print("Preview Tile Layers (for Leaflet)")
    print("="*60)
    for layer in layers:
        print(f"\n{layer['name']}:")
        print(f"  URL: {layer['tile_url']}")
        print(f"  Attribution: {layer['attribution']}")
    
    # Save to JSON
    preview_json = {
        "bbox": bbox,
        "years": years,
        "sensors": sensors,
        "max_cloud": max_cloud,
        "layers": layers
    }
    with open("preview_layers.json", "w") as f:
        json.dump(preview_json, f, indent=2)
    print("\n✓ Saved to preview_layers.json")
    
    return layers


def export_dataset(bbox, years, sensors, output_dir, max_cloud=20, scale=10):
    """Export MNDWI dataset to GeoTIFFs."""
    import gee
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Exporting Dataset")
    print("="*60)
    print(f"Output directory: {output_path.resolve()}")
    
    # Export Sentinel-2 MNDWI
    if 's2' in sensors:
        print("\nExporting Sentinel-2 MNDWI scenes...")
        s2_dir = output_path / "sentinel2"
        s2_dir.mkdir(exist_ok=True)
        
        try:
            paths = gee.export_s2_mndwi(
                bbox, years[0], years[1], 
                str(s2_dir), max_cloud=max_cloud, scale=scale
            )
            print(f"✓ Exported {len(paths)} Sentinel-2 scenes to {s2_dir}")
        except Exception as e:
            print(f"⚠ Sentinel-2 export failed: {e}")
    
    # For Landsat, we'd need the full pipeline modules
    # For now, just note this in the output
    if 'landsat' in sensors:
        print("\nNote: Landsat export requires the full DeltaSAT pipeline modules.")
        print("      Use: MODULES_DIR=/path/to/modules MOCK_MODE=0 python app.py")
        landsat_dir = output_path / "landsat"
        landsat_dir.mkdir(exist_ok=True)
        (landsat_dir / "README.txt").write_text(
            "Landsat export: Run via the full DeltaSAT pipeline with MODULES_DIR set.\n"
        )
    
    # Create a metadata file
    metadata = {
        "bbox": bbox,
        "years": list(years),
        "sensors": sensors,
        "max_cloud": max_cloud,
        "scale": scale,
        "export_date": str(Path.cwd().resolve()),
        "directories": {
            "sentinel2": str(s2_dir) if 's2' in sensors else None,
            "landsat": str(landsat_dir) if 'landsat' in sensors else None
        }
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved to {metadata_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"Bounding box: {bbox}")
    print(f"Years: {years[0]} to {years[1]}")
    print(f"Sensors: {sensors}")
    print(f"Max cloud cover: {max_cloud}%")
    print(f"Output directory: {output_path.resolve()}")
    
    if 's2' in sensors:
        s2_files = list(s2_dir.rglob("*.tif"))
        print(f"Sentinel-2 files: {len(s2_files)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Quick script to create cloud-ready datasets for DeltaSAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--bbox', type=str, 
                        help='Bounding box as "w,s,e,n" or "w s e n"')
    parser.add_argument('--years', type=str,
                        help='Year range as "2020,2022" or "2020-2022"')
    parser.add_argument('--sensors', type=str, default='both',
                        choices=['landsat', 's2', 'both'],
                        help='Sensor type (default: both)')
    parser.add_argument('--output', type=str, default='./cloud_dataset',
                        help='Output directory (default: ./cloud_dataset)')
    parser.add_argument('--max-cloud', type=int, default=20,
                        help='Max cloud cover percentage (default: 20)')
    parser.add_argument('--scale', type=int, default=10,
                        help='Export scale in meters (default: 10)')
    parser.add_argument('--preview-only', action='store_true',
                        help='Generate preview layers only, no export')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive input (default if no args)')
    
    args = parser.parse_args()
    
    # If no arguments provided, use interactive mode
    if not any(vars(args).values()) or args.interactive:
        bbox, years, sensors, output_dir, max_cloud, preview_only = get_user_input()
    else:
        if not args.bbox:
            print("Error: --bbox is required for non-interactive mode")
            parser.print_help()
            sys.exit(1)
        if not args.years:
            print("Error: --years is required for non-interactive mode")
            parser.print_help()
            sys.exit(1)
        
        bbox = parse_bbox(args.bbox)
        years = parse_years(args.years)
        
        if args.sensors == 'both':
            sensors = ['landsat', 's2']
        elif args.sensors == 's2':
            sensors = ['s2']
        else:
            sensors = ['landsat']
        
        output_dir = args.output
        max_cloud = args.max_cloud
        preview_only = args.preview_only
    
    # Initialize GEE
    print("\nInitializing Google Earth Engine...")
    init_ee()
    
    # Generate preview
    print("\n" + "="*60)
    print("DeltaSAT Cloud Dataset Generation")
    print("="*60)
    
    if preview_only:
        layers = generate_preview(bbox, years, sensors, max_cloud)
        print("\n✓ Preview layers generated (no export performed)")
    else:
        # Do both preview and export
        layers = generate_preview(bbox, years, sensors, max_cloud)
        export_dataset(bbox, years, sensors, output_dir, max_cloud, args.scale)
        print("\n✓ Dataset creation complete!")


if __name__ == "__main__":
    main()
