#!/usr/bin/env python3
"""Merge clean 2025 ICDL state masks into three monthly Corn Belt GeoTIFFs.

Run in the existing corn-yield environment:
    python merge_cornbelt_icdl_2025_monthly.py

Requires only numpy and rasterio (already used by the production script).
Uses aligned native pixels, with no reprojection or resampling. Codes remain
0=non-corn, 1=corn, 255=NoData. Gaps are NOT filled. First valid state wins
in an overlap; disagreements are counted. State inputs are never modified.
Completed outputs with unchanged input paths/sizes/mtimes are skipped on rerun.
Use --overwrite to rebuild; --overviews adds optional mode overview pyramids.
Only run one instance at a time. An interrupted month restarts on the next run.
"""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

STATES = ('IA', 'IL', 'IN', 'NE', 'MN', 'MO', 'KS', 'SD', 'ND', 'OH', 'WI', 'MI')
MONTHS = ('June', 'July', 'August')
VERSION = '1'


def integer_offset(value):
    nearest = round(value)
    if abs(value - nearest) > 1e-5:
        raise ValueError('State rasters are not aligned to the same pixel grid.')
    return int(nearest)


def inspect_grid(sources):
    anchor = sources[0].transform
    offsets = []
    for src in sources:
        t = src.transform
        if src.crs != rasterio.crs.CRS.from_epsg(5070):
            raise ValueError('Expected EPSG:5070: ' + src.name)
        if src.count != 1 or src.dtypes[0] != 'uint8' or src.nodata != 255:
            raise ValueError('Expected single-band uint8, NoData=255: ' + src.name)
        if not np.allclose([t.a, t.b, t.d, t.e], [30, 0, 0, -30], atol=1e-8, rtol=0):
            raise ValueError('Expected north-up 30 m grid: ' + src.name)
        offsets.append((integer_offset((t.c-anchor.c)/30),
                        integer_offset((anchor.f-t.f)/30)))
    min_c = min(c for c, r in offsets)
    min_r = min(r for c, r in offsets)
    width = max(c + s.width for (c, r), s in zip(offsets, sources)) - min_c
    height = max(r + s.height for (c, r), s in zip(offsets, sources)) - min_r
    transform = anchor * Affine.translation(min_c, min_r)
    offsets = [(c-min_c, r-min_r) for c, r in offsets]
    return transform, width, height, offsets


def fingerprint(paths, overviews):
    payload = dict(version=VERSION, overviews=overviews, sources=[
        (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def merge_month(paths, output, month, block_size=2048, overwrite=False, overviews=False):
    signature = fingerprint(paths, overviews)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in paths]
        transform, width, height, offsets = inspect_grid(sources)
        if output.exists() and not overwrite:
            with rasterio.open(output) as existing:
                tags = existing.tags()
                if (tags.get('merge_signature') == signature
                        and tags.get('merge_complete') == 'YES'
                        and existing.shape == (height, width)
                        and existing.transform == transform):
                    print('  Already complete, unchanged inputs: ' + str(output), flush=True)
                    return json.loads(tags['merge_summary'])
            raise RuntimeError('Existing output differs or is unverified. Use --overwrite: ' + str(output))

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.stem + '.partial.tif')
        # This name is reserved for this script's unfinished output.
        if partial.exists():
            partial.unlink()
        profile = dict(driver='GTiff', width=width, height=height, count=1,
                       crs=sources[0].crs, transform=transform, dtype='uint8',
                       nodata=255, tiled=True, blockxsize=512, blockysize=512,
                       compress='DEFLATE', BIGTIFF='YES')
        total = ((width+block_size-1)//block_size)*((height+block_size-1)//block_size)
        print('  Grid: {:,} x {:,}; uncompressed base band: {:.2f} GiB'.format(
            width, height, width*height/1024**3), flush=True)
        print('  Writing ' + str(output), flush=True)
        counts = np.zeros(256, dtype=np.int64)
        conflicts = 0
        done = 0
        started = time.monotonic()
        last_report = started
        with rasterio.open(partial, 'w', **profile) as dst:
            for row in range(0, height, block_size):
                h = min(block_size, height-row)
                for col in range(0, width, block_size):
                    w = min(block_size, width-col)
                    data = np.full((h, w), 255, dtype=np.uint8)
                    for src, (sc, sr) in zip(sources, offsets):
                        left, right = max(col, sc), min(col+w, sc+src.width)
                        top, bottom = max(row, sr), min(row+h, sr+src.height)
                        if left >= right or top >= bottom:
                            continue
                        values = src.read(1, window=Window(left-sc, top-sr, right-left, bottom-top))
                        valid = (values == 0) | (values == 1)
                        if np.any(~valid & (values != 255)):
                            raise ValueError('Unexpected class values (expected 0/1/255): ' + src.name)
                        view = data[top-row:bottom-row, left-col:right-col]
                        conflicts += int(np.count_nonzero(valid & (view != 255) & (view != values)))
                        take = valid & (view == 255)
                        view[take] = values[take]
                    dst.write(data, 1, window=Window(col, row, w, h))
                    counts += np.bincount(data.ravel(), minlength=256)
                    done += 1
                    now = time.monotonic()
                    if done == 1 or done == total or now-last_report >= 15:
                        print('  {:5.1f}% | {}/{} blocks | {:.1f} min'.format(
                            100*done/total, done, total, (now-started)/60), flush=True)
                        last_report = now
            if overviews:
                print('  Building mode overviews...', flush=True)
                levels = [n for n in (2, 4, 8, 16) if min(width, height)//n >= 1]
                if levels:
                    dst.build_overviews(levels, Resampling.mode)
                    dst.update_tags(ns='rio_overview', resampling='mode')
            summary = dict(Month=month, States=len(paths), Width=width, Height=height,
                           Corn_pixels=int(counts[1]), Noncorn_pixels=int(counts[0]),
                           NoData_pixels=int(counts[255]),
                           Corn_km2=round(int(counts[1])*0.0009, 4),
                           Valid_km2=round(int(counts[0]+counts[1])*0.0009, 4),
                           Overlap_conflict_comparisons=conflicts, Path=str(output))
            dst.update_tags(merge_signature=signature, merge_complete='YES',
                            merge_summary=json.dumps(summary),
                            source_files=json.dumps([str(p) for p in paths]),
                            class_codes='0=non-corn;1=corn;255=NoData',
                            overlap_rule='first valid state in source_files order')
        # Do not publish a partial write under the final filename.
        with rasterio.open(partial) as check:
            if check.shape != (height, width) or check.tags().get('merge_complete') != 'YES':
                raise RuntimeError('Output verification failed: ' + str(partial))
        os.replace(partial, output)
        print('  DONE | corn: {:,.2f} km2 | valid: {:,.2f} km2 | overlap conflicts: {:,}'.format(
            summary['Corn_km2'], summary['Valid_km2'], conflicts), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(r'D:\corn_yield'))
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--block-size', type=int, default=2048)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--overviews', action='store_true')
    args = parser.parse_args()
    if args.block_size < 256 or args.block_size > 4096:
        parser.error('--block-size must be between 256 and 4096')
    base = args.root / ('Our_ICDL_{}_CornBelt'.format(args.year))
    source_root = base / ('states_clean_rebuilt_{}'.format(args.year))
    output_root = base / ('cornbelt_monthly_mosaics_{}'.format(args.year))
    inputs = {month: [source_root / month / state /
              ('Our{}{}_{}_CornMask30m_CLEAN_REBUILT.tif'.format(month, args.year, state))
              for state in STATES] for month in MONTHS}
    missing = [str(p) for paths in inputs.values() for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required state mosaics missing:\n' + '\n'.join(missing))
    print('Checking all 36 state rasters before merging...', flush=True)
    for paths in inputs.values():
        with ExitStack() as stack:
            inspect_grid([stack.enter_context(rasterio.open(p)) for p in paths])
    print('Inputs passed grid checks. Values: 0=non-corn, 1=corn, 255=NoData.', flush=True)
    print('NoData includes outside-state areas and existing gaps; it is not a state coverage percentage.', flush=True)
    rows = []
    # Bound GDAL's additional cache to 256 MiB; the merge itself is block-based.
    with rasterio.Env(GDAL_CACHEMAX=256*1024*1024):
        for month in MONTHS:
            print('\n' + month.upper(), flush=True)
            output = output_root / ('Our{}{}_CornBelt_CornMask30m_CLEAN_REBUILT.tif'.format(month, args.year))
            rows.append(merge_month(inputs[month], output, month, args.block_size,
                                    args.overwrite, args.overviews))
    summary_path = output_root / ('CornBelt_{}_monthly_mosaic_summary.csv'.format(args.year))
    temporary_csv = summary_path.with_suffix('.partial.csv')
    with temporary_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, summary_path)
    print('\nALL THREE MONTHLY MOSAICS COMPLETE', flush=True)
    print(str(output_root), flush=True)
    print('Summary: ' + str(summary_path), flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Rerun to skip completed months and restart the unfinished month.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('\nERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
