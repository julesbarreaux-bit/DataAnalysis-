import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sps
from data import df_merged, col_groups, col_limits, GROUP_UNITS

_HAS_DATE = 'target_shipped_date' in df_merged.columns

# Wafers compared as one group ("Selected") against every other wafer ("Others")
SELECTED_WAFERS = [
    'O26J18-835-T03', 'O26E30-744-T03', 'O26E30-745-T03', 'O26E31-746-T03',
    'O26E31-747-T03', 'O26E31-748-T03', 'O26E31-750-T03', 'O26E31-751-T03',
    'O26J18-836-T03', 'O26E31-752-T03', 'O26E31-754-T03', 'O26E31-755-T03',
    'O26E31-756-T03', 'O26E31-757-T03', 'O26E31-758-T03', 'O26F01-759-T03',
    'O26F01-760-T03', 'O26F01-761-T03', 'O26F01-762-T03', 'O26F01-763-T03',
]

SEL, OTH = 'Selected', 'Others'
POPULATIONS = [SEL, OTH]

# Minimum smaller-group size below which a two-sample test is flagged as underpowered
UNDERPOWERED_N = 20

_IS_SEL = df_merged['wafer_id'].isin(SELECTED_WAFERS)


def _make_labels(cols: list) -> list:
    """
    Build clean display labels from full column names.

    Rules (applied per group, so context-aware):
      1. Always strip the group prefix (parts[0]).
      2. If every column has ≥ 3 parts AND the middle part (parts[1]) is
         identical across all columns → it carries no information, drop it
         and show only parts[2:].
      3. Otherwise show all remaining parts joined with ' | '.
      4. Clean up deduplication suffixes: _2 → (2), _3 → (3), etc.
    """
    if not cols:
        return []

    split = [c.split(' | ') for c in cols]

    def _clean(s: str) -> str:
        return re.sub(r'_(\d+)$', r' (\1)', s)

    all_tri = all(len(p) >= 3 for p in split)
    uniform_mid = all_tri and len({p[1] for p in split}) == 1

    labels = []
    for parts in split:
        if len(parts) == 1:
            labels.append(_clean(parts[0]))
        elif len(parts) == 2:
            labels.append(_clean(parts[1]))
        elif uniform_mid:
            labels.append(' | '.join(_clean(p) for p in parts[2:]))
        else:
            labels.append(' | '.join(_clean(p) for p in parts[1:]))

    return labels


class WaferModel:
    def __init__(self):
        self._stats_cache:   dict = {}
        self._raw_cache:     dict = {}
        self._compare_cache: pd.DataFrame | None = None
        self._dates: dict = {p: (None, None) for p in POPULATIONS}

    def groups(self) -> list:
        return [g for g in col_groups if g != 'Unknown']

    def unit(self, group: str) -> str:
        raw = GROUP_UNITS.get(group, '')
        return 'µm' if raw == 'um' else raw

    def date_bounds(self) -> dict:
        """Returns {'Selected': (min_ts, max_ts), 'Others': (min_ts, max_ts)}"""
        result = {}
        if not _HAS_DATE:
            return result
        for pop, mask in [(SEL, _IS_SEL), (OTH, ~_IS_SEL)]:
            dates = df_merged.loc[mask, 'target_shipped_date'].dropna()
            if len(dates) >= 2 and dates.min() != dates.max():
                result[pop] = (dates.min(), dates.max())
            elif len(dates) >= 1:
                d = dates.iloc[0]
                result[pop] = (d - pd.Timedelta(days=30), d + pd.Timedelta(days=30))
            else:
                result[pop] = (None, None)
        return result

    def set_date_filter(self, ranges: dict):
        """ranges: {'Selected': (start, end), 'Others': (start, end)}"""
        for pop in POPULATIONS:
            self._dates[pop] = ranges.get(pop, (None, None))
        self._stats_cache.clear()
        self._raw_cache.clear()
        self._compare_cache = None

    def stats(self, group: str) -> pd.DataFrame:
        if group not in self._stats_cache:
            self._stats_cache[group] = self._build_stats(group)
        return self._stats_cache[group]

    def raw(self, group: str) -> dict:
        if group not in self._raw_cache:
            self._raw_cache[group] = self._build_raw(group)
        return self._raw_cache[group]

    def limits(self, group: str) -> dict:
        num_cols = self._num_cols(group)
        labels   = _make_labels(num_cols)
        return {
            lbl: col_limits[col]
            for col, lbl in zip(num_cols, labels)
            if col in col_limits
        }

    def compare_tests(self) -> pd.DataFrame:
        """
        For every measurement column, test whether the distribution differs
        between the Selected wafers and all the other wafers, using the
        non-parametric Mann-Whitney U test (no normality assumption).
        Comparisons where the smaller group has fewer than UNDERPOWERED_N
        points are flagged as underpowered.

        p_adj is the Benjamini-Hochberg adjusted p-value, computed over all
        tested measurements at once to control the false discovery rate.
        """
        if self._compare_cache is not None:
            return self._compare_cache

        m_sel, m_other = self._mask(SEL), self._mask(OTH)

        rows = []
        for group in self.groups():
            cols = self._num_cols(group)
            for col, label in zip(cols, _make_labels(cols)):
                ssl = df_merged.loc[m_sel,   col].dropna()
                sot = df_merged.loc[m_other, col].dropna()

                vs = self._mw_test(sot, ssl)

                rows.append({
                    'group': group, 'label': label,
                    'n_sel': len(ssl), 'n_other': len(sot),
                    'test':    vs['test'],
                    'p_value': vs['p_value'],
                    'note':    vs['note'],
                })

        df = pd.DataFrame(rows)
        df['p_adj'] = np.nan
        tested = df['p_value'].notna()
        if tested.any():
            df.loc[tested, 'p_adj'] = sps.false_discovery_control(df.loc[tested, 'p_value'])

        self._compare_cache = df
        return self._compare_cache

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mw_test(ref: pd.Series, grp: pd.Series) -> dict:
        """Two-sided Mann-Whitney U test (non-parametric, no normality assumption)."""
        n_ref, n_grp = len(ref), len(grp)
        if n_ref < 2 or n_grp < 2:
            return {'test': None, 'p_value': np.nan, 'note': 'Insufficient data'}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, p = sps.mannwhitneyu(ref, grp, alternative='two-sided')
        test = 'Mann-Whitney U'

        note = f'Underpowered (n<{UNDERPOWERED_N})' if min(n_ref, n_grp) < UNDERPOWERED_N else ''
        return {'test': test, 'p_value': float(p), 'note': note}

    def _mask(self, pop: str):
        m = _IS_SEL if pop == SEL else ~_IS_SEL
        if _HAS_DATE:
            s, e = self._dates[pop]
            if s is not None:
                m = m & (df_merged['target_shipped_date'] >= s)
            if e is not None:
                m = m & (df_merged['target_shipped_date'] <= e)
        return m

    def _num_cols(self, group: str) -> list:
        cols = [c for c in col_groups.get(group, []) if c in df_merged.columns]
        return df_merged[cols].select_dtypes(include='number').columns.tolist()

    def _build_stats(self, group: str) -> pd.DataFrame:
        num_cols = self._num_cols(group)
        labels   = _make_labels(num_cols)
        m_sel, m_oth = self._mask(SEL), self._mask(OTH)
        rows = []
        for col, label in zip(num_cols, labels):
            ssl = df_merged.loc[m_sel, col].dropna()
            sot = df_merged.loc[m_oth, col].dropna()
            rows.append({
                'label':      label,
                'mean_sel':   ssl.mean()   if len(ssl) else np.nan,
                'median_sel': ssl.median() if len(ssl) else np.nan,
                'var_sel':    ssl.var()    if len(ssl) >= 2 else np.nan,
                'n_sel':      int(len(ssl)),
                'mean_oth':   sot.mean()   if len(sot) else np.nan,
                'median_oth': sot.median() if len(sot) else np.nan,
                'n_oth':      int(len(sot)),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _build_raw(self, group: str) -> dict:
        num_cols = self._num_cols(group)
        labels   = _make_labels(num_cols)
        m_sel, m_oth = self._mask(SEL), self._mask(OTH)
        d_sel = [df_merged.loc[m_sel, c].dropna().to_numpy() for c in num_cols]
        d_oth = [df_merged.loc[m_oth, c].dropna().to_numpy() for c in num_cols]
        return {'labels': labels, 'data_sel': d_sel, 'data_oth': d_oth}
