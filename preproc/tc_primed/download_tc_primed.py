#!/usr/bin/env python3
"""
download_tcprimed.py
--------------------

Bulk-download files from the NOAA TC-PRIMED public S3 bucket.

Examples
========
# 1)  All 2015 North-Atlantic storms
python download_tcprimed.py \
    --prefix v01r01/final/2015/AL/ \
    --dest   /scratch/$USER/tcprimed

# 2)  Everything in v01r01/final (≈1.6 TB – be sure you really want it!)
python download_tcprimed.py \
    --prefix v01r01/final/ \
    --workers 32 \
    --dest   /scratch/$USER/tcprimed
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

BUCKET_NAME = "noaa-nesdis-tcprimed-pds"


def list_keys(prefix: str):
    """
    Recursively list object keys under `prefix` in the public bucket.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def download_one(s3, key: str, size: int, dest_root: str):
    """
    Download a single object unless it already exists locally at the same size.
    """
    local_path = os.path.join(dest_root, os.path.relpath(key, start=prefix))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if file exists and the size matches
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        return

    s3.download_file(BUCKET_NAME, key, local_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TC-PRIMED data (public S3).")
    parser.add_argument(
        "--prefix", required=True, help="S3 prefix to pull (e.g. 'v01r01/final/2015/AL/')."
    )
    parser.add_argument("--dest", required=True, help="Destination directory on the cluster.")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel download threads (default: 8)."
    )
    args = parser.parse_args()

    prefix = args.prefix.lstrip("/")  # ensure no leading slash
    dest_root = os.path.abspath(args.dest)

    # Anonymous (unsigned) S3 client
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Find everything we’re going to grab
    objects = list(list_keys(prefix))
    total_size = sum(size for _, size in objects)

    print(f"Found {len(objects):,} files – {total_size/1e9:,.2f} GB.")

    # Multi-threaded download with progress bar
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_one, s3_client, key, size, dest_root) for key, size in objects
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            pass

    print("Done.")
