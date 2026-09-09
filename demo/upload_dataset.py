#!/usr/bin/env python3
"""
upload_dataset.py - Quick script to upload datasets to cloud storage.

Supports:
- Google Cloud Storage (GCS) - recommended for GEE integration
- Local file system organization for Oracle Cloud upload

Usage:
    # Upload to Google Cloud Storage
    python upload_dataset.py --source ./my_dataset --target gs://my-bucket/datasets

    # Organize for Oracle Cloud (creates tar.gz for upload)
    python upload_dataset.py --source ./my_dataset --oracle --output oracle_upload.tar.gz

    # Interactive mode
    python upload_dataset.py

Requirements:
    # For GCS
    pip install google-cloud-storage
    gcloud auth application-default login  # or set GOOGLE_APPLICATION_CREDENTIALS
    
    # For Oracle
    pip install oci  # optional, for direct Oracle Object Storage upload
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload DeltaSAT datasets to cloud storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--source', type=str, 
                        help='Source directory containing dataset files')
    parser.add_argument('--target', type=str,
                        help='Target cloud path (e.g., gs://bucket/path or oracle://bucket/namespace)')
    parser.add_argument('--gcs', action='store_true',
                        help='Upload to Google Cloud Storage')
    parser.add_argument('--oracle', action='store_true',
                        help='Package for Oracle Object Storage (creates tar.gz)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for oracle package (default: <source>.tar.gz)')
    parser.add_argument('--include', type=str, nargs='*', default=['*.tif', '*.geojson', '*.json', '*.csv'],
                        help='File patterns to include (default: GeoTIFFs, JSON, CSV)')
    parser.add_argument('--exclude', type=str, nargs='*', default=['__pycache__', '.DS_Store'],
                        help='Directories to exclude')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive input')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test run without actual upload')
    
    return parser.parse_args()


def get_user_input():
    """Interactive input collection."""
    print("=" * 60)
    print("DeltaSAT Cloud Upload Tool")
    print("=" * 60)
    
    source = input("Source directory: ").strip()
    if not source or not Path(source).exists():
        print(f"Error: Directory '{source}' does not exist")
        sys.exit(1)
    
    print("\nUpload destination:")
    print("  1. Google Cloud Storage (GCS)")
    print("  2. Oracle Object Storage (package as tar.gz)")
    choice = input("Choose (1 or 2, default: 1): ").strip()
    
    if choice == '2':
        target = None
        oracle = True
        output = input("Output tar.gz path (default: <source>.tar.gz): ").strip()
        if not output:
            output = f"{source}.tar.gz"
    else:
        target = input("GCS path (e.g., gs://my-bucket/datasets): ").strip()
        oracle = False
        output = None
    
    dry_run = input("Dry run? (y/n, default: n): ").strip().lower()
    dry_run = dry_run in ['y', 'yes']
    
    return source, target, oracle, output, dry_run


def collect_files(source_dir, include_patterns, exclude_dirs):
    """Collect all files matching patterns, excluding specified directories."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    files = []
    for pattern in include_patterns:
        # Convert glob pattern to regex-like matching
        for ext in ['*.tif', '*.geojson', '*.json', '*.csv', '*.png']:
            if pattern == ext or pattern.endswith(ext[1:]):
                files.extend(source_path.rglob(ext))
    
    # Remove excluded directories
    filtered_files = []
    for f in files:
        if any(excl in f.parts for excl in exclude_dirs):
            continue
        filtered_files.append(f)
    
    # Remove duplicates and sort
    unique_files = list(set(filtered_files))
    unique_files.sort()
    
    return unique_files


def upload_to_gcs(source_dir, target_path, files, dry_run=False):
    """Upload files to Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        print("Error: google-cloud-storage not installed")
        print("Install with: pip install google-cloud-storage")
        sys.exit(1)
    
    # Parse GCS path
    if not target_path.startswith('gs://'):
        print(f"Error: GCS path must start with 'gs://', got: {target_path}")
        sys.exit(1)
    
    # Extract bucket and prefix
    parts = target_path[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    print(f"\nUploading to GCS:")
    print(f"  Bucket: {bucket_name}")
    print(f"  Prefix: {prefix}")
    print(f"  Files: {len(files)}")
    
    if dry_run:
        print("\n[DRY RUN] Would upload:")
        for f in files[:10]:
            rel = f.relative_to(source_dir)
            gcs_path = f"gs://{bucket_name}/{prefix}/{rel.as_posix()}" if prefix else f"gs://{bucket_name}/{rel.as_posix()}"
            print(f"  {f} -> {gcs_path}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        return
    
    # Initialize client
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        print("\nMake sure you have:")
        print("  1. Installed google-cloud-storage: pip install google-cloud-storage")
        print("  2. Authenticated: gcloud auth application-default login")
        print("  3. Or set GOOGLE_APPLICATION_CREDENTIALS env var")
        sys.exit(1)
    
    # Upload files
    uploaded = 0
    errors = 0
    source_path = Path(source_dir)
    
    for file_path in files:
        try:
            rel_path = file_path.relative_to(source_path)
            gcs_blob_name = f"{prefix}/{rel_path.as_posix()}" if prefix else rel_path.as_posix()
            blob = bucket.blob(gcs_blob_name)
            
            # Create directory structure
            blob.upload_from_filename(str(file_path))
            uploaded += 1
            
            if uploaded % 10 == 0:
                print(f"  Uploaded: {uploaded}/{len(files)}")
                
        except Exception as e:
            print(f"  Error uploading {file_path}: {e}")
            errors += 1
    
    print(f"\n✓ Uploaded {uploaded} files to gs://{bucket_name}/{prefix}")
    if errors:
        print(f"  ⚠ {errors} files failed")
    
    # Save metadata
    metadata = {
        "source": str(source_path.resolve()),
        "target": target_path,
        "uploaded_at": datetime.now().isoformat(),
        "files": [str(f.relative_to(source_path)) for f in files],
        "count": uploaded,
        "errors": errors
    }
    
    metadata_path = source_path / "upload_metadata_gcs.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Upload metadata saved to {metadata_path}")
    
    return uploaded, errors


def package_for_oracle(source_dir, output_path, files, dry_run=False):
    """Package files into a tar.gz for Oracle Object Storage upload."""
    source_path = Path(source_dir)
    output_path = Path(output_path)
    
    print(f"\nPackaging for Oracle Object Storage:")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_path}")
    print(f"  Files: {len(files)}")
    
    if dry_run:
        print("\n[DRY RUN] Would package:")
        for f in files[:10]:
            print(f"  {f.relative_to(source_path)}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        return
    
    # Create tar.gz
    try:
        with tarfile.open(str(output_path), "w:gz") as tar:
            for file_path in files:
                arcname = file_path.relative_to(source_path)
                tar.add(str(file_path), arcname=str(arcname))
        
        print(f"✓ Created {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Save metadata
        metadata = {
            "source": str(source_path.resolve()),
            "output": str(output_path.resolve()),
            "packaged_at": datetime.now().isoformat(),
            "files": [str(f.relative_to(source_path)) for f in files],
            "count": len(files),
            "size_bytes": output_path.stat().st_size
        }
        
        metadata_path = source_path / "upload_metadata_oracle.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Package metadata saved to {metadata_path}")
        
        # Print Oracle upload commands
        print("\n" + "="*60)
        print("Oracle Object Storage Upload Commands")
        print("="*60)
        print("\nOption 1: Using OCI CLI")
        print(f"  oci os object put -bn <bucket-name> --file {output_path}")
        print("\nOption 2: Using Python (oci package)")
        print("  pip install oci")
        print(f"  python -c \"import oci; print('See Oracle docs for upload script')\"")
        
        return len(files), 0
        
    except Exception as e:
        print(f"Error creating package: {e}")
        return 0, 1


def upload_to_oracle_direct(source_dir, target_path, files, dry_run=False):
    """Direct upload to Oracle Object Storage (if oci package available)."""
    try:
        import oci
    except ImportError:
        print("\noci package not installed for direct Oracle upload")
        print("Install with: pip install oci")
        print("Or use the packaged tar.gz with OCI CLI")
        return 0, 0
    
    # Parse oracle://bucket/namespace/path
    if not target_path.startswith('oracle://'):
        print(f"Error: Oracle path must start with 'oracle://', got: {target_path}")
        return 0, 1
    
    parts = target_path[8:].split('/', 2)
    bucket = parts[0]
    namespace = parts[1] if len(parts) > 1 else None
    prefix = parts[2] if len(parts) > 2 else ''
    
    if not namespace:
        namespace = input("Oracle namespace: ").strip()
    
    print(f"\nUploading to Oracle Object Storage:")
    print(f"  Bucket: {bucket}")
    print(f"  Namespace: {namespace}")
    print(f"  Prefix: {prefix}")
    
    if dry_run:
        print("\n[DRY RUN] Would upload:")
        for f in files[:10]:
            rel = f.relative_to(source_dir)
            print(f"  {rel}")
        return
    
    # This would require proper OCI configuration
    print("\nNote: Direct Oracle upload requires OCI configuration")
    print("See: https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/pythonsdk.htm")
    print("\nUsing package mode instead...")
    
    return 0, 0


def main():
    args = parse_args()
    
    # Interactive mode if no source provided
    if not args.source or args.interactive:
        source, target, oracle, output, dry_run = get_user_input()
    else:
        source = args.source
        target = args.target
        oracle = args.oracle
        output = args.output
        dry_run = args.dry_run
    
    if not source:
        print("Error: --source is required")
        sys.exit(1)
    
    # Collect files
    print(f"\nScanning {source}...")
    files = collect_files(source, args.include, args.exclude)
    print(f"Found {len(files)} files matching patterns: {args.include}")
    
    if not files:
        print("No files found to upload. Check your include/exclude patterns.")
        sys.exit(1)
    
    # Handle GCS upload
    if args.gcs or (target and target.startswith('gs://')):
        if not target:
            target = input("GCS path (e.g., gs://my-bucket/datasets): ").strip()
        if not target:
            print("Error: GCS path required")
            sys.exit(1)
        upload_to_gcs(source, target, files, dry_run or args.dry_run)
    
    # Handle Oracle
    elif oracle or (target and target.startswith('oracle://')):
        if oracle and not output:
            output = f"{source}.tar.gz"
        elif not output:
            output = f"{source}.tar.gz"
        package_for_oracle(source, output, files, dry_run or args.dry_run)
    
    # Default: if target looks like GCS, use that
    elif target and target.startswith('gs://'):
        upload_to_gcs(source, target, files, dry_run or args.dry_run)
    
    # No clear target, show options
    else:
        print("\n" + "="*60)
        print("Upload Options")
        print("="*60)
        print("\n1. Google Cloud Storage (Recommended for GEE integration)")
        print("   Run: python upload_dataset.py --source <dir> --target gs://bucket/path --gcs")
        print("\n2. Oracle Object Storage (Package as tar.gz)")
        print("   Run: python upload_dataset.py --source <dir> --oracle")
        print("\n3. Interactive mode")
        print("   Run: python upload_dataset.py --interactive")


if __name__ == "__main__":
    main()
