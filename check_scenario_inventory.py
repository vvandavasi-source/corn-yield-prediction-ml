#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_scenario_inventory.py

Run this on YOUR machine (Windows) to see exactly what MODIS scenario files
you have vs. what CY2_COMPACT_PRISM_CDL_ICDL_2022_2025_DUAL_ALTERNATIVE.py
requires in Cell 18.

It does three things:
  1. Finds every CSV whose name contains "MODIS".
  2. Tests each against the SAME strict regex the model script uses, so you
     see which files the script would actually accept.
  3. Prints a year x scenario coverage grid (2022-2025) for the strict match,
     PLUS a loose "what do I seem to have" breakdown so you can map your real
     files onto the required slots.

No pandas / no heavy deps -- just the standard library.

EDIT THE PATHS BELOW, then:  python check_scenario_inventory.py
"""

import os
import re
import glob
from collections import defaultdict

# ------------------------------------------------------------------
# EDIT THESE to match your machine
# ------------------------------------------------------------------
SEARCH_ROOTS = [
    r"C:\Users\Patron\Downloads\Author's ICDL",
    r"C:\Users\Patron\Downloads\Author's ICDL\scenarios",
    r"C:\Users\Patron\Downloads\Author's ICDL\corn_yield_model_work\data",
]

YEARS = [2022, 2023, 2024, 2025]
MONTHS = ["JUNE", "JULY", "AUGUST"]

# ------------------------------------------------------------------
# The EXACT contract the model script enforces (copied from the script)
# ------------------------------------------------------------------
SCENARIO_PATTERN = re.compile(
    r"^CORN_MODIS_(2022|2023|2024|2025)_"
    r"(PREVIOUS_CDL|FINAL_CDL|"
    r"CLEAN_REBUILT_(JUNE|JULY|AUGUST)|"
    r"STUDY_(JUNE|JULY|AUGUST)|"
    r"OUR_(JUNE|JULY|AUGUST))"
    r"\.CSV$",
    re.IGNORECASE,
)


def normalize_filename(path):
    """Same normalization the model script uses."""
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)   # strip " (1)", " (2)" browser dupes
    return (stem + ext).upper()


def strict_slot(norm_name):
    """Return (year, slot) if the script would accept this file, else None."""
    m = SCENARIO_PATTERN.match(norm_name)
    if not m:
        return None
    year = int(m.group(1))
    token = m.group(2).upper()
    if token == "PREVIOUS_CDL":
        return year, "Previous CDL"
    if token == "FINAL_CDL":
        return year, "Final CDL"
    for fam in ("CLEAN_REBUILT_", "OUR_"):
        if token.startswith(fam):
            return year, "ICDL " + token.replace(fam, "").title()
    if token.startswith("STUDY_"):
        return year, "Published " + token.replace("STUDY_", "").title()
    return None


def loose_guess(norm_name):
    """Best-effort guess of (year, family, month) for files that DON'T match."""
    ym = re.search(r"(2022|2023|2024|2025)", norm_name)
    year = int(ym.group(1)) if ym else None
    month = next((mo for mo in MONTHS if mo in norm_name
                  or mo[:3] in norm_name), None)
    fams = []
    for kw, fam in [
        ("PREVIOUS", "previous"), ("PREV", "previous"),
        ("FINAL", "final"),
        ("ICDL", "icdl"), ("CLEAN", "icdl/clean"), ("OUR", "our"),
        ("CROPSMART", "cropsmart"), ("STUDY", "study"),
        ("PUBLISHED", "published"),
    ]:
        if kw in norm_name and fam not in fams:
            fams.append(fam)
    return year, (",".join(fams) if fams else "?"), (month or "-")


# ------------------------------------------------------------------
# Scan
# ------------------------------------------------------------------
candidates = []
for root in SEARCH_ROOTS:
    if not os.path.exists(root):
        print(f"[note] search root does not exist: {root}")
        continue
    candidates.extend(
        glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    )

candidates = sorted(set(os.path.abspath(p) for p in candidates))
modis = [p for p in candidates if "MODIS" in normalize_filename(p)]

print("=" * 78)
print(f"CSV files found:            {len(candidates)}")
print(f"...with 'MODIS' in name:    {len(modis)}")
print("=" * 78)

strict_hits = {}         # (year, slot) -> path
unmatched = []
for p in modis:
    norm = normalize_filename(p)
    hit = strict_slot(norm)
    if hit:
        strict_hits.setdefault(hit, p)
    else:
        unmatched.append(p)

# ------------------------------------------------------------------
# 1) STRICT coverage grid = what the model script will actually accept
# ------------------------------------------------------------------
slots = ["Previous CDL", "ICDL June", "ICDL July", "ICDL August", "Final CDL"]
print("\nSTRICT COVERAGE (files the model script will accept):")
header = "  year  | " + " | ".join(f"{s:<12}" for s in slots)
print(header)
print("  " + "-" * (len(header) - 2))
for y in YEARS:
    cells = []
    for s in slots:
        cells.append(" OK " if (y, s) in strict_hits else "  . ")
    print(f"  {y}  |  " + "  |  ".join(f"{c:<10}" for c in cells))
print("\n  (OK = present and correctly named;  . = missing or misnamed)")

runnable = [y for y in YEARS
            if any((y, f"ICDL {m.title()}") in strict_hits for m in MONTHS)
            and (y, "Previous CDL") in strict_hits]
print(f"\n  Years the primary A-vs-B test could run as-is: "
      f"{runnable if runnable else 'NONE'}")

# ------------------------------------------------------------------
# 2) LOOSE breakdown = what you actually seem to have (any naming)
# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("LOOSE BREAKDOWN of every MODIS csv (helps map real files -> slots):")
print("=" * 78)
by_year = defaultdict(list)
for p in modis:
    y, fam, month = loose_guess(normalize_filename(p))
    by_year[y].append((fam, month, os.path.basename(p)))
for y in sorted(by_year, key=lambda v: (v is None, v)):
    print(f"\n  {y if y is not None else 'year=?'}:")
    for fam, month, base in sorted(by_year[y]):
        print(f"    family={fam:<16} month={month:<8} {base}")

# ------------------------------------------------------------------
# 3) Files the script will silently ignore
# ------------------------------------------------------------------
if unmatched:
    print("\n" + "=" * 78)
    print(f"{len(unmatched)} MODIS file(s) the script will NOT recognize "
          f"(these are why it raises FileNotFoundError):")
    print("=" * 78)
    for p in unmatched:
        print("   " + os.path.basename(p))

print("\nDone. Paste this output back and we'll decide: rename/remap vs. "
      "regenerate the masked MODIS aggregations.")