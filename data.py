import re
import numpy as np
import pandas as pd

EXCEL_FILE = 'I-PEX U-LEAP V2.xlsx'


GROUP_SHORT_NAMES = {
    "PZT Layer Thickness [um] (RF-02)": "PZT Layer Thickness",
    "Front-side geometric distances - Mask Alignment [um] (RF-03)": "Front-side Mask Alignment",
    "Back-side geometric distances\nMask Alignment [um] (RF-04)": "Back-side Mask Alignment",
    "Piston-Size [um]": "Piston Size",
    "Back cavity-Size [um]\nBTM_DRIE Side": "Back Cavity Size",
    "Si thickness [um] (RF-10)": "Si Thickness",
    "Bow [um] (RF-11)": "Bow",
    "Scribe Line - WF Ra [nm] (RF-22)": "Scribe Line Ra",
    "Z-offset (RF-16)": "Z-offset",
    "IPS internal electrical check (only one cantilever on the TEG are checked)\n Note: If the cantilever to be measured is not suitable for measurement for some reason, the measurement will be done on a chip in the nearby area.\nIf the average value of 10 leakage measurements per WF is less than \"7.50E+08 Ω\", the product will be shipped after pre-notification.\n　*Specification: 1.25E+08 Ω (Since IPS measures only 1 cantilever out of 6, the spec multiplied by 6 is used as the pre-notification criteria.)": "IPS Electrical Check",
    "PVI thickness / um": "PVI Thickness",
}


def _extract_unit(name: str) -> str:
    m = re.search(r'\[([^\]]+)\]', name)
    if m:
        return m.group(1)
    m = re.search(r'[/\\]\s*(\w+)\s*(?:\n|$)', name)
    if m:
        return m.group(1)
    return ''


GROUP_UNITS: dict[str, str] = {
    short: _extract_unit(orig)
    for orig, short in GROUP_SHORT_NAMES.items()
}


def _build_functional_column_names(raw_header: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """
    Combine the header rows into readable names: Section [| Direction] | Sub | Position.
    Row 1 (direction) is included only when it is a single letter such as X or Y.
    It is forward-filled *within* each r0 group only — preventing values from leaking
    into subsequent groups (e.g. Front-side Y bleeding into Back-side columns).
    Also returns col_groups: short_group_name → [col_names].
    """
    r0_raw = raw_header.iloc[0]          # raw (not ffilled) — used for group-boundary detection
    r0     = r0_raw.ffill()              # ffilled — used for column name building
    r1_raw = raw_header.iloc[1]
    r2_raw = raw_header.iloc[2]
    r3     = raw_header.iloc[3]
    r4     = raw_header.iloc[4] if len(raw_header) > 4 else pd.Series(dtype=object)  # 8" radii for PZT

    # Forward-fill r1 AND r2 within each r0 group only (reset at every group boundary).
    # This prevents values from one group leaking into the next via pandas global ffill.
    r1_carry = None
    r2_carry = None
    r1_vals, r2_vals = [], []
    for i in range(len(r3)):
        if pd.notna(r0_raw.iloc[i]):     # new group → reset both carries
            r1_carry = None
            r2_carry = None
        if pd.notna(r1_raw.iloc[i]):
            r1_carry = r1_raw.iloc[i]
        if pd.notna(r2_raw.iloc[i]):
            r2_carry = r2_raw.iloc[i]
        r1_vals.append(r1_carry)
        r2_vals.append(r2_carry)
    r1 = pd.Series(r1_vals, dtype=object)
    r2 = pd.Series(r2_vals, dtype=object)

    seen: dict[str, int] = {}
    names = []
    col_groups: dict[str, list[str]] = {}

    for i in range(len(r3)):
        parts = []
        if pd.notna(r0[i]):
            parts.append(str(r0[i]).strip())

        # Include r1 whenever it carries useful sub-group info (X/Y direction,
        # process-step label, etc.). Newlines are collapsed to ' / '.
        r1_val = str(r1[i]).replace('\n', ' / ').strip() if pd.notna(r1[i]) else ''
        if r1_val and r1_val not in parts:
            parts.append(r1_val)

        if pd.notna(r2[i]) and str(r2[i]).strip() not in parts:
            parts.append(str(r2[i]).strip())
        if pd.notna(r3[i]):
            r3_val = str(r3[i]).strip()
            r4_val = str(r4.iloc[i]).strip() if i < len(r4) and pd.notna(r4.iloc[i]) else ''
            label  = f'{r3_val} / {r4_val}' if r4_val else r3_val
            if label not in parts:
                parts.append(label)
        name = " | ".join(parts) if parts else f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)

        # Map to short group name (always from r0)
        raw_group = str(r0[i]).strip() if pd.notna(r0[i]) else "Unknown"
        short = next(
            (v for k, v in GROUP_SHORT_NAMES.items() if raw_group.startswith(k[:40])),
            raw_group,
        )
        col_groups.setdefault(short, []).append(name)

    return names, col_groups


# --- FunctionalTests ---
_raw_header = pd.read_excel(EXCEL_FILE, sheet_name='FunctionalTests', header=None, nrows=5)
_col_names, col_groups = _build_functional_column_names(_raw_header)

# Upper/Lower Limit rows sit at rows 5–6 (before the data which starts at row 7)
_raw_limits = pd.read_excel(EXCEL_FILE, sheet_name='FunctionalTests', header=None, skiprows=5, nrows=2)
col_limits: dict[str, dict] = {}
for _i, _cname in enumerate(_col_names):
    if _i >= len(_raw_limits.columns):
        break
    _u = pd.to_numeric(_raw_limits.iloc[0, _i], errors='coerce')
    _l = pd.to_numeric(_raw_limits.iloc[1, _i], errors='coerce')
    _uv = float(_u) if pd.notna(_u) else None
    _lv = float(_l) if pd.notna(_l) else None
    if _uv is not None or _lv is not None:
        col_limits[_cname] = {'upper': _uv, 'lower': _lv}

df_functional = pd.read_excel(EXCEL_FILE, sheet_name='FunctionalTests', header=None, skiprows=7)
df_functional.columns = _col_names
df_functional = df_functional[df_functional['Wafer ID'].notna()].reset_index(drop=True)

np_functional = df_functional.to_numpy()

# --- Combined Data View (full, unfiltered) ---
df_combined = pd.read_excel(EXCEL_FILE, sheet_name='Combined Data View')

np_combined = df_combined.to_numpy()

# --- Merge on Wafer ID (full, unfiltered) ---
df_merged = pd.merge(
    df_combined,
    df_functional,
    left_on='wafer_id',
    right_on='Wafer ID',
    how='inner'
).drop(columns=['Wafer ID'])

np_merged = df_merged.to_numpy()


def get_merged_for_months(months: int) -> pd.DataFrame:
    """Return df_merged filtered to the last `months` months by target_shipped_date."""
    cutoff = pd.Timestamp.today() - pd.DateOffset(months=months)
    df_c = df_combined[df_combined['target_shipped_date'] >= cutoff]
    return pd.merge(
        df_c, df_functional,
        left_on='wafer_id', right_on='Wafer ID',
        how='inner'
    ).drop(columns=['Wafer ID'])
