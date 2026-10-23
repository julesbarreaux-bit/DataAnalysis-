"""
View layer — all tkinter / matplotlib widgets.
No business logic; receives data dicts/DataFrames from the controller.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from model import UNDERPOWERED_N

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = '#FFFFFF'
BG_ALT  = '#F8F9FA'
BG_NAV  = '#1A1D23'
SEP     = '#E2E4E7'
TEXT    = '#1A1D23'
MUTED   = '#737B8C'
ACCENT  = '#0066CC'
CLIM    = '#D63031'   # spec limit lines

SEL, OTH = 'Selected', 'Others'

# Chart – bars
C_OTH_M  = '#0052CC'   # Others mean
C_OTH_MD = '#5B9CF6'   # Others median
C_SEL_M  = '#4A3AA7'   # Selected mean
C_SEL_MD = '#9B8FE0'   # Selected median

POP_COLORS = {SEL: C_SEL_M, OTH: C_OTH_M}

# Chart – distribution (Others as box plot, Selected wafers as dots)
BOX_FACE   = '#E8F0FE'
BOX_LINE   = '#0052CC'
SEL_MARKER = '^'

# Statistical significance
SIG_ALPHA = 0.05
SIG_BG    = '#F5A623'   # darker highlight: BH-adjusted p passes the threshold
SIG_RAW_BG = '#FDE7C0'  # light highlight: only the raw p passes the threshold

FNT      = 'Segoe UI'
FNT_SZ   = 9
FNT_SZ_T = 13


def _fmt(v) -> str:
    if pd.isna(v):
        return '—'
    a = abs(v)
    if a >= 10000: return f'{v:,.0f}'
    if a >= 100:   return f'{v:,.1f}'
    if a >= 1:     return f'{v:.3f}'
    if a >= 0.001: return f'{v:.4f}'
    return f'{v:.3e}'


# ── Range slider (dual-handle canvas widget) ─────────────────────────────────
class RangeSlider(tk.Canvas):
    _H  = 28
    _TY = 14
    _TH = 4
    _HR = 7
    _PAD = 12

    def __init__(self, parent, min_val: float, max_val: float,
                 on_change, color: str,
                 init_low: float = None, init_high: float = None, **kw):
        kw.update(height=self._H, bg=BG_ALT, highlightthickness=0, bd=0)
        super().__init__(parent, **kw)
        self._min = min_val
        self._max = max_val if max_val > min_val else min_val + 1
        self._low  = init_low  if init_low  is not None else min_val
        self._high = init_high if init_high is not None else self._max
        self._on_change = on_change
        self._color = color
        self._drag  = None
        self.bind('<Configure>',      lambda _: self._redraw())
        self.bind('<ButtonPress-1>',  self._on_press)
        self.bind('<B1-Motion>',      self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _tw(self) -> float:
        return max(self.winfo_width() - 2 * self._PAD, 1)

    def _to_x(self, val: float) -> float:
        return self._PAD + (val - self._min) / (self._max - self._min) * self._tw()

    def _to_val(self, x: float) -> float:
        v = (x - self._PAD) / self._tw() * (self._max - self._min) + self._min
        return max(self._min, min(self._max, v))

    def _redraw(self):
        self.delete('all')
        xl, xh = self._to_x(self._low), self._to_x(self._high)
        ty = self._TY
        # track background
        self.create_rectangle(self._PAD, ty - self._TH // 2,
                               self._PAD + self._tw(), ty + self._TH // 2,
                               fill=SEP, outline='')
        # active range
        self.create_rectangle(xl, ty - self._TH // 2, xh, ty + self._TH // 2,
                               fill=self._color, outline='')
        # handles
        for x in (xl, xh):
            self.create_oval(x - self._HR, ty - self._HR,
                             x + self._HR, ty + self._HR,
                             fill='white', outline=self._color, width=2)

    def _on_press(self, event):
        xl, xh = self._to_x(self._low), self._to_x(self._high)
        self._drag = 'low' if abs(event.x - xl) <= abs(event.x - xh) else 'high'

    def _on_drag(self, event):
        val = self._to_val(event.x)
        if self._drag == 'low':
            self._low = min(val, self._high)
        else:
            self._high = max(val, self._low)
        self._redraw()

    def _on_release(self, event):
        self._drag = None
        self._on_change(self._low, self._high)

    def get(self):
        return self._low, self._high


# ── Date filter bar (one range slider per wafer population) ───────────────────
class DateFilterBar(tk.Frame):
    def __init__(self, parent, bounds: dict, on_change, **kw):
        super().__init__(parent, bg=BG_ALT, **kw)
        self._on_change = on_change
        self._sliders:  dict[str, RangeSlider] = {}
        self._lbl_lo:   dict[str, tk.Label]    = {}
        self._lbl_hi:   dict[str, tk.Label]    = {}

        inner = tk.Frame(self, bg=BG_ALT)
        inner.pack(fill='x', padx=16, pady=4)
        inner.columnconfigure(2, weight=1)

        # Global date axis so both sliders share the same time scale
        valid = [(lo, hi) for lo, hi in bounds.values()
                 if lo is not None and not pd.isna(lo)]
        if not valid:
            return
        global_min = float(min(lo for lo, hi in valid).toordinal())
        global_max = float(max(hi for lo, hi in valid).toordinal())

        row_idx = 0
        for size, (lo, hi) in bounds.items():
            color = POP_COLORS.get(size, ACCENT)
            if lo is None or pd.isna(lo):
                continue
            self._build_row(inner, row_idx, size, color, lo, hi, global_min, global_max)
            row_idx += 1

        tk.Frame(self, bg=SEP, height=1).pack(fill='x', side='bottom')

    def _build_row(self, parent, row: int, size: str, color: str,
                   lo: pd.Timestamp, hi: pd.Timestamp,
                   global_min: float, global_max: float):
        init_lo = float(lo.toordinal())
        init_hi = float(hi.toordinal())

        tk.Label(parent, text=size, bg=BG_ALT, fg=color,
                 font=(FNT, 7, 'bold'), width=8, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 4))

        lbl_lo = tk.Label(parent, text=lo.strftime('%Y-%m-%d'),
                          bg=BG_ALT, fg=MUTED, font=(FNT, 7), width=11, anchor='e')
        lbl_lo.grid(row=row, column=1, sticky='e', padx=(0, 4))

        slider = RangeSlider(
            parent, global_min, global_max,
            on_change=lambda l, h, sz=size: self._changed(sz, l, h),
            color=color,
            init_low=init_lo, init_high=init_hi,
        )
        slider.grid(row=row, column=2, sticky='ew', padx=4)

        lbl_hi = tk.Label(parent, text=hi.strftime('%Y-%m-%d'),
                          bg=BG_ALT, fg=MUTED, font=(FNT, 7), width=11, anchor='w')
        lbl_hi.grid(row=row, column=3, sticky='w', padx=(4, 0))

        self._sliders[size] = slider
        self._lbl_lo[size]  = lbl_lo
        self._lbl_hi[size]  = lbl_hi

    def _changed(self, size: str, lo: float, hi: float):
        fmt = '%Y-%m-%d'
        self._lbl_lo[size].config(text=pd.Timestamp.fromordinal(int(lo)).strftime(fmt))
        self._lbl_hi[size].config(text=pd.Timestamp.fromordinal(int(hi)).strftime(fmt))
        self._on_change()

    def get_ranges(self) -> dict:
        result = {}
        for size, slider in self._sliders.items():
            lo, hi = slider.get()
            result[size] = (pd.Timestamp.fromordinal(int(lo)),
                            pd.Timestamp.fromordinal(int(hi)))
        return result


# ── Scroll-to-zoom helper ─────────────────────────────────────────────────────
def _add_scroll_zoom(ax, canvas):
    def on_scroll(event):
        if event.inaxes != ax:
            return
        scale = 0.8 if event.button == 'up' else 1.25
        ylim = ax.get_ylim()
        cy = event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2
        ax.set_ylim(cy - (cy - ylim[0]) * scale, cy + (ylim[1] - cy) * scale)
        canvas.draw_idle()
    canvas.mpl_connect('scroll_event', on_scroll)


# ── Scrollable matplotlib canvas ──────────────────────────────────────────────
class ScrollChart(tk.Frame):
    """Wraps a matplotlib Figure in a horizontally-scrollable container."""

    def __init__(self, parent, fig: Figure, px_w: int, px_h: int, **kw):
        super().__init__(parent, bg=BG, **kw)

        hbar = ttk.Scrollbar(self, orient='horizontal')
        hbar.pack(side='bottom', fill='x')

        cv = tk.Canvas(self, bg=BG, xscrollcommand=hbar.set, highlightthickness=0)
        cv.pack(fill='both', expand=True)
        hbar.config(command=cv.xview)

        inner = tk.Frame(cv, bg=BG)
        cv.create_window((0, 0), window=inner, anchor='nw')

        self.mpl = FigureCanvasTkAgg(fig, master=inner)
        self.mpl.draw()
        self.mpl.get_tk_widget().config(width=px_w, height=px_h)
        self.mpl.get_tk_widget().pack()

        inner.update_idletasks()
        cv.config(scrollregion=(0, 0, px_w, px_h))


# ── Navbar ────────────────────────────────────────────────────────────────────
class NavBar(tk.Frame):
    def __init__(self, parent, groups: list, on_select, **kw):
        super().__init__(parent, bg=BG_NAV, **kw)
        self._btns: dict[str, tk.Label] = {}
        self._active = None
        self._on_select = on_select

        # Brand
        tk.Label(self, text='WAFER  ·  Selected vs Others',
                 bg=BG_NAV, fg='#FFFFFF',
                 font=(FNT, 10, 'bold'),
                 padx=20, pady=0).pack(side='left')

        # Thin vertical separator
        tk.Frame(self, bg='#353A45', width=1).pack(side='left', fill='y',
                                                    pady=8, padx=4)

        # Tab buttons
        for g in groups:
            short = g if len(g) <= 21 else g[:19] + '…'
            lbl = tk.Label(
                self, text=short,
                bg=BG_NAV, fg='#9DA5B4',
                font=(FNT, FNT_SZ),
                padx=12, pady=0, cursor='hand2',
            )
            lbl.bind('<Button-1>', lambda e, grp=g: self._on_select(grp))
            lbl.bind('<Enter>',   lambda e, l=lbl: l.config(fg='#FFFFFF')
                     if l != self._btns.get(self._active) else None)
            lbl.bind('<Leave>',   lambda e, l=lbl, g2=g: l.config(
                fg='#FFFFFF' if g2 == self._active else '#9DA5B4'))
            lbl.pack(side='left', ipady=10)
            self._btns[g] = lbl

        # Bottom border
        tk.Frame(parent, bg=SEP, height=1).pack(side='top', fill='x')

    def set_active(self, group: str):
        if self._active and self._active in self._btns:
            self._btns[self._active].config(
                bg=BG_NAV, fg='#9DA5B4', relief='flat')
        self._active = group
        if group in self._btns:
            self._btns[group].config(bg='#252A34', fg='#FFFFFF', relief='flat')


# ── Statistics table ──────────────────────────────────────────────────────────
class StatsTable(tk.Frame):
    def __init__(self, parent, df: pd.DataFrame, **kw):
        super().__init__(parent, bg=BG, **kw)

        s = ttk.Style()
        s.configure('WT.Treeview',
                    background=BG, fieldbackground=BG,
                    foreground=TEXT, font=(FNT, FNT_SZ), rowheight=22)
        s.configure('WT.Treeview.Heading',
                    background=BG_ALT, foreground=TEXT,
                    font=(FNT, FNT_SZ, 'bold'), relief='flat')
        s.map('WT.Treeview',
              background=[('selected', '#DBEAFE')],
              foreground=[('selected', TEXT)])

        cols = ['Measurement',
                'Others Mean', 'Others Median', 'n (Others)',
                'Sel. Mean', 'Sel. Median', 'Sel. Variance', 'n (Sel.)',
                'Δ Mean', 'Δ Median']

        tree = ttk.Treeview(self, columns=cols, show='headings',
                            height=min(len(df), 7), style='WT.Treeview')
        widths = {
            'Measurement': 260,
            'Others Mean': 95, 'Others Median': 95, 'n (Others)': 75,
            'Sel. Mean': 95, 'Sel. Median': 95, 'Sel. Variance': 100, 'n (Sel.)': 65,
            'Δ Mean': 100, 'Δ Median': 100,
        }
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=widths.get(c, 80),
                        anchor='w' if c == 'Measurement' else 'center',
                        minwidth=40)

        tree.tag_configure('odd',  background=BG_ALT)
        tree.tag_configure('even', background=BG)

        for i, row in df.iterrows():
            # Δ = Selected − Others (NaN propagates to '—')
            dm = row['mean_sel']   - row['mean_oth']
            dd = row['median_sel'] - row['median_oth']
            vals = [row['label'],
                    _fmt(row['mean_oth']), _fmt(row['median_oth']), str(row['n_oth']),
                    _fmt(row['mean_sel']), _fmt(row['median_sel']),
                    _fmt(row['var_sel']), str(row['n_sel']),
                    _fmt(dm), _fmt(dd)]

            tree.insert('', 'end', values=vals,
                        tags=('odd' if i % 2 else 'even',))

        vsb = ttk.Scrollbar(self, orient='vertical',   command=tree.yview)
        hsb = ttk.Scrollbar(self, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)


# ── Bar chart (mean & median) ─────────────────────────────────────────────────
class BarChart(tk.Frame):
    def __init__(self, parent, df: pd.DataFrame,
                 limits: dict = None, ylabel: str = 'Value', **kw):
        super().__init__(parent, bg=BG, **kw)

        n     = len(df)
        fig_w = max(6.0, n * 1.3 + 1.5)
        fig_h = 8.0
        px_w  = int(fig_w * 96)
        px_h  = int(fig_h * 96)

        fig = Figure(figsize=(fig_w, fig_h), dpi=96, facecolor=BG)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG)

        x = np.arange(n)

        w = 0.17
        ax.bar(x - 1.5*w, df['mean_oth'],   w, color=C_OTH_M,  label='Others Mean',   zorder=3)
        ax.bar(x - 0.5*w, df['median_oth'], w, color=C_OTH_MD, label='Others Median', zorder=3)
        ax.bar(x + 0.5*w, df['mean_sel'],   w, color=C_SEL_M,  label='Selected Mean',   zorder=3)
        ax.bar(x + 1.5*w, df['median_sel'], w, color=C_SEL_MD, label='Selected Median', zorder=3)

        # Lock y-axis to data range; limits drawn outside won't break the view
        ax.autoscale(enable=True, axis='y', tight=False)
        _ymin, _ymax = ax.get_ylim()

        # ── Spec limit lines ──────────────────────────────────────────────────
        added_upper = added_lower = False
        if limits:
            span = 0.44
            for xi, lbl in zip(x, df['label'].tolist()):
                if lbl not in limits:
                    continue
                lim = limits[lbl]
                if lim['upper'] is not None:
                    kw_l = {'label': 'Upper Limit'} if not added_upper else {}
                    ax.hlines(lim['upper'], xi - span, xi + span,
                              colors=CLIM, linewidth=1.5, linestyle='--', zorder=7, **kw_l)
                    added_upper = True
                if lim['lower'] is not None:
                    kw_l = {'label': 'Lower Limit'} if not added_lower else {}
                    ax.hlines(lim['lower'], xi - span, xi + span,
                              colors=CLIM, linewidth=1.5, linestyle=':', zorder=7, **kw_l)
                    added_lower = True

        ax.set_ylim(_ymin, _ymax)   # restore data-only range after drawing limits

        ax.set_xticks(x)
        ax.set_xticklabels(df['label'].tolist(), rotation=35, ha='right',
                           fontsize=7.5, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
        ax.tick_params(axis='y', labelsize=7.5, colors=MUTED)
        ax.tick_params(axis='x', length=0)
        leg = ax.legend(fontsize=8, frameon=True, framealpha=1,
                        edgecolor=SEP, facecolor=BG)
        leg.get_frame().set_linewidth(0.5)
        ax.grid(axis='y', color=SEP, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(SEP)
        fig.tight_layout(pad=1.2)

        chart = ScrollChart(self, fig, px_w, px_h)
        chart.pack(fill='both', expand=True)
        _add_scroll_zoom(ax, chart.mpl)

        tb = NavigationToolbar2Tk(chart.mpl, self)
        tb.update()
        tb.config(background=BG)
        tb.pack(side='bottom', fill='x')


# ── Distribution chart (Others box plot + Selected scatter) ──────────────────
class DistChart(tk.Frame):
    """Box plot of the Others distribution, individual dots for Selected wafers."""

    def __init__(self, parent, raw: dict, limits: dict = None, ylabel: str = 'Value', **kw):
        super().__init__(parent, bg=BG, **kw)

        labels   = raw['labels']
        data_oth = raw['data_oth']
        data_sel = raw['data_sel']
        n        = len(labels)

        rng   = np.random.default_rng(42)
        fig_w = max(5.5, n * 0.85 + 1.5)
        fig_h = 5.5
        px_w  = int(fig_w * 96)
        px_h  = int(fig_h * 96)

        fig = Figure(figsize=(fig_w, fig_h), dpi=96, facecolor=BG)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG)

        x = np.arange(1, n + 1)

        # ── Box plots for Others ─────────────────────────────────────────────
        valid = [i for i, d in enumerate(data_oth) if len(d) >= 3]
        if valid:
            ax.boxplot(
                [data_oth[i] for i in valid],
                positions=[x[i] for i in valid],
                widths=0.45,
                patch_artist=True,
                notch=False,
                boxprops=dict(facecolor=BOX_FACE, color=BOX_LINE, linewidth=0.9),
                medianprops=dict(color=BOX_LINE, linewidth=1.6),
                whiskerprops=dict(color=BOX_LINE, linewidth=0.9, linestyle='--'),
                capprops=dict(color=BOX_LINE, linewidth=0.9),
                flierprops=dict(marker='.', color=BOX_LINE,
                                markersize=3, alpha=0.35, linestyle='none'),
                zorder=3,
            )

        # ── Scatter for the Selected wafers ───────────────────────────────────
        for i, ds in enumerate(data_sel):
            if len(ds) == 0:
                continue
            jitter = rng.uniform(-0.14, 0.14, len(ds))
            ax.scatter(
                x[i] + jitter, ds,
                color=C_SEL_M, marker=SEL_MARKER, s=40, zorder=6,
                alpha=0.85, linewidths=0.6, edgecolors='white',
            )

        # Lock y-axis to data range; limits drawn outside won't break the view
        ax.autoscale(enable=True, axis='y', tight=False)
        _ymin, _ymax = ax.get_ylim()

        # ── Spec limit lines ──────────────────────────────────────────────────
        added_upper = added_lower = False
        if limits:
            span = 0.38
            for xi, lbl in zip(x, labels):
                if lbl not in limits:
                    continue
                lim = limits[lbl]
                if lim['upper'] is not None:
                    kw_l = {'label': 'Upper Limit'} if not added_upper else {}
                    ax.hlines(lim['upper'], xi - span, xi + span,
                              colors=CLIM, linewidth=1.5, linestyle='--', zorder=7, **kw_l)
                    added_upper = True
                if lim['lower'] is not None:
                    kw_l = {'label': 'Lower Limit'} if not added_lower else {}
                    ax.hlines(lim['lower'], xi - span, xi + span,
                              colors=CLIM, linewidth=1.5, linestyle=':', zorder=7, **kw_l)
                    added_lower = True

        ax.set_ylim(_ymin, _ymax)   # restore data-only range after drawing limits

        # Legend handles
        from matplotlib.patches import Patch
        import matplotlib.lines as mlines
        legend_handles = [
            Patch(facecolor=BOX_FACE, edgecolor=BOX_LINE, linewidth=0.9,
                  label='Others distribution'),
        ]
        if any(len(d) > 0 for d in data_sel):
            legend_handles.append(
                mlines.Line2D([], [], color=C_SEL_M, marker=SEL_MARKER,
                              linestyle='none', markersize=7, label='Selected wafers')
            )
        if added_upper:
            legend_handles.append(
                mlines.Line2D([], [], color=CLIM, linewidth=1.5, linestyle='--',
                              label='Upper Limit')
            )
        if added_lower:
            legend_handles.append(
                mlines.Line2D([], [], color=CLIM, linewidth=1.5, linestyle=':',
                              label='Lower Limit')
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right',
                           fontsize=7.5, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
        ax.tick_params(axis='y', labelsize=7.5, colors=MUTED)
        ax.tick_params(axis='x', length=0)
        leg = ax.legend(handles=legend_handles, fontsize=8, frameon=True,
                        framealpha=1, edgecolor=SEP, facecolor=BG)
        leg.get_frame().set_linewidth(0.5)
        ax.grid(axis='y', color=SEP, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(SEP)
        ax.set_xlim(0.3, n + 0.7)
        fig.tight_layout(pad=1.2)

        chart = ScrollChart(self, fig, px_w, px_h)
        chart.pack(fill='both', expand=True)
        _add_scroll_zoom(ax, chart.mpl)

        tb = NavigationToolbar2Tk(chart.mpl, self)
        tb.update()
        tb.config(background=BG)
        tb.pack(side='bottom', fill='x')


# ── Section header helper ─────────────────────────────────────────────────────
def _section(parent, title: str) -> tk.Frame:
    tk.Label(parent, text=title.upper(),
             bg=BG, fg=MUTED,
             font=(FNT, 7, 'bold'),
             anchor='w', padx=16).pack(fill='x', pady=(10, 2))
    tk.Frame(parent, bg=SEP, height=1).pack(fill='x', padx=16)
    body = tk.Frame(parent, bg=BG)
    body.pack(fill='both', expand=True, padx=16, pady=(6, 0))
    return body


# ── Group page ────────────────────────────────────────────────────────────────
class GroupPage(tk.Frame):
    """Full content area for one metric group."""

    def __init__(self, parent, group: str,
                 df: pd.DataFrame, raw: dict, limits: dict = None,
                 unit: str = '', **kw):
        super().__init__(parent, bg=BG, **kw)

        if df.empty:
            tk.Label(self, text='No numeric data for this group.',
                     bg=BG, fg=MUTED,
                     font=(FNT, 12)).pack(expand=True)
            return

        n_oth = int(df['n_oth'].max())
        n_sel = int(df['n_sel'].max())

        # ── Page header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x', padx=16, pady=(14, 0))

        tk.Label(hdr, text=group,
                 bg=BG, fg=TEXT,
                 font=(FNT, FNT_SZ_T, 'bold'),
                 anchor='w').pack(side='left')

        badge_frame = tk.Frame(hdr, bg=BG)
        badge_frame.pack(side='right')
        _badge(badge_frame, f'n  Others  =  {n_oth}', C_OTH_M)
        _badge(badge_frame, f'n  Selected  =  {n_sel}', C_SEL_M)

        tk.Frame(self, bg=SEP, height=1).pack(fill='x', padx=16, pady=(10, 0))

        # ── Statistics table ─────────────────────────────────────────────────
        tbl_hdr = _section(self, 'Statistics')
        StatsTable(tbl_hdr, df).pack(fill='x')

        tk.Frame(self, bg=SEP, height=1).pack(fill='x', padx=16, pady=(10, 0))

        # ── Charts (side by side) ─────────────────────────────────────────────
        charts_hdr = _section(self, 'Charts')
        charts_hdr.pack_configure(expand=True, fill='both')

        charts_row = tk.Frame(charts_hdr, bg=BG)
        charts_row.pack(fill='both', expand=True)
        charts_row.columnconfigure(0, weight=55, uniform='charts')
        charts_row.columnconfigure(1, weight=1)   # separator
        charts_row.columnconfigure(2, weight=45, uniform='charts')
        charts_row.rowconfigure(0, weight=1)

        ylabel = f"{group} ({unit})" if unit else group

        # Left: bar chart
        bar_wrap = tk.Frame(charts_row, bg=BG)
        bar_wrap.grid(row=0, column=0, sticky='nsew', padx=(0, 0))
        _chart_title(bar_wrap, 'Mean  ·  Median  comparison')
        BarChart(bar_wrap, df, limits=limits, ylabel=ylabel).pack(fill='both', expand=True)

        # Vertical separator
        tk.Frame(charts_row, bg=SEP, width=1).grid(
            row=0, column=1, sticky='ns', pady=4)

        # Right: distribution chart
        dist_wrap = tk.Frame(charts_row, bg=BG)
        dist_wrap.grid(row=0, column=2, sticky='nsew', padx=(0, 0))
        _chart_title(dist_wrap, 'Others  distribution  +  Selected  wafers')
        DistChart(dist_wrap, raw, limits=limits, ylabel=ylabel).pack(fill='both', expand=True)


# ── Vertical scroll container ─────────────────────────────────────────────────
class VScroll(tk.Frame):
    """Vertically-scrollable container. Pack child widgets into .body."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        vbar = ttk.Scrollbar(self, orient='vertical')
        vbar.pack(side='right', fill='y')

        canvas = tk.Canvas(self, bg=BG, yscrollcommand=vbar.set, highlightthickness=0)
        canvas.pack(side='left', fill='both', expand=True)
        vbar.config(command=canvas.yview)

        self.body = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=self.body, anchor='nw')

        self.body.bind('<Configure>',
                       lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win, width=e.width))

        def _wheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), 'units')
        canvas.bind('<MouseWheel>', _wheel)
        self.body.bind('<MouseWheel>', _wheel)


def _group_title(parent, name: str):
    row = tk.Frame(parent, bg=BG)
    row.pack(fill='x', pady=(0, 6))
    tk.Frame(row, bg=ACCENT, width=4).pack(side='left', fill='y')
    tk.Label(row, text=name, bg=BG, fg=TEXT, font=(FNT, 11, 'bold'),
             anchor='w', padx=8).pack(side='left', fill='y')


# ── Statistical test table (one measurement group, headers included) ─────────
class StatTestGroupTable(tk.Frame):
    """Selected wafers vs all other wafers, Mann-Whitney U test per measurement."""

    def __init__(self, parent, df: pd.DataFrame, **kw):
        super().__init__(parent, bg=BG, **kw)

        s = ttk.Style()
        s.configure('STT.Treeview',
                    background=BG, fieldbackground=BG,
                    foreground=TEXT, font=(FNT, FNT_SZ), rowheight=22)
        s.configure('STT.Treeview.Heading',
                    background=BG_ALT, foreground=TEXT,
                    font=(FNT, FNT_SZ, 'bold'), relief='flat')
        s.map('STT.Treeview',
              background=[('selected', '#DBEAFE')],
              foreground=[('selected', TEXT)])

        cols = ['Measurement', 'n (Sel · Other)',
                'Test', 'p-value', 'p-adj (BH)', 'Threshold (α)', 'Result']

        tree = ttk.Treeview(self, columns=cols, show='headings',
                            height=len(df), style='STT.Treeview')
        widths = {
            'Measurement': 260, 'n (Sel · Other)': 110,
            'Test': 130,
            'p-value': 90, 'p-adj (BH)': 90, 'Threshold (α)': 90, 'Result': 260,
        }
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=widths.get(c, 90),
                        anchor='w' if c == 'Measurement' else 'center',
                        minwidth=40)

        tree.tag_configure('odd',  background=BG_ALT)
        tree.tag_configure('even', background=BG)
        tree.tag_configure('sig',  background=SIG_BG)
        tree.tag_configure('sig_raw', background=SIG_RAW_BG)

        def _result(p, p_adj, test, note):
            if test is None or pd.isna(p):
                return 'Insufficient data'
            if p_adj < SIG_ALPHA:
                verdict = 'Significant (after BH)'
            elif p < SIG_ALPHA:
                verdict = 'Borderline (raw p only)'
            else:
                verdict = 'Not significant'
            return f'{verdict} ({note})' if note else verdict

        for i, (_, row) in enumerate(df.iterrows()):
            n_str  = f'{row["n_sel"]} · {row["n_other"]}'
            tested = row['test'] is not None and pd.notna(row['p_value'])
            vals = [
                row['label'], n_str,
                row['test'] or 'n/a', _fmt(row['p_value']), _fmt(row['p_adj']), f'{SIG_ALPHA}',
                _result(row['p_value'], row['p_adj'], row['test'], row['note']),
            ]
            if tested and row['p_adj'] < SIG_ALPHA:
                tag = 'sig'
            elif tested and row['p_value'] < SIG_ALPHA:
                tag = 'sig_raw'
            else:
                tag = 'odd' if i % 2 else 'even'
            tree.insert('', 'end', values=vals, tags=(tag,))

        tree.pack(fill='x')


# ── Statistical test page ─────────────────────────────────────────────────────
class StatTestPage(tk.Frame):
    """Full content area for the 'Statistical Tests' tab, one table per measurement group."""

    def __init__(self, parent, df: pd.DataFrame, **kw):
        super().__init__(parent, bg=BG, **kw)

        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x', padx=16, pady=(14, 0))
        tk.Label(hdr, text='Statistical Tests',
                 bg=BG, fg=TEXT, font=(FNT, FNT_SZ_T, 'bold'),
                 anchor='w').pack(side='left')

        tk.Frame(self, bg=SEP, height=1).pack(fill='x', padx=16, pady=(10, 0))

        intro = _section(self, 'Method')
        intro.pack_configure(expand=False)
        tk.Label(intro, text=(
            'For every measurement, the Selected wafers are compared against all the other wafers '
            '(within the date windows set above). '
            'The test is the non-parametric Mann-Whitney U (two-sided), which makes no normality '
            'assumption and is robust to skewed distributions and outliers. Because many measurements are '
            'tested at once, p-adj (BH) is the Benjamini-Hochberg adjusted p-value over all tests, which '
            f'limits false positives. Darker amber: p-adj < {SIG_ALPHA} (solid result). Light amber: only the '
            f'raw p < {SIG_ALPHA} (borderline, may be chance). Comparisons where the '
            f'smaller side has fewer than {UNDERPOWERED_N} points are flagged "Underpowered": a non-significant '
            'result there usually means "not enough data", not "no real difference".'
        ), bg=BG, fg=MUTED, font=(FNT, 8), anchor='w', justify='left',
           wraplength=1500).pack(fill='x', pady=(0, 4))

        tk.Frame(self, bg=SEP, height=1).pack(fill='x', padx=16, pady=(10, 0))

        if df.empty:
            tk.Label(self, text='No numeric data available.',
                     bg=BG, fg=MUTED, font=(FNT, 12)).pack(expand=True)
            return

        tk.Label(self, text='RESULTS  ·  ONE TABLE PER MEASUREMENT GROUP', bg=BG, fg=MUTED,
                 font=(FNT, 7, 'bold'), anchor='w', padx=16).pack(fill='x', pady=(10, 2))

        scroll = VScroll(self)
        scroll.pack(fill='both', expand=True, padx=16, pady=(0, 10))

        for group in df['group'].unique():
            sub = df[df['group'] == group].reset_index(drop=True)
            if sub.empty:
                continue
            _group_title(scroll.body, group)
            StatTestGroupTable(scroll.body, sub).pack(fill='x')
            tk.Frame(scroll.body, bg=SEP, height=1).pack(fill='x', pady=(14, 14))


def _badge(parent, text: str, color: str):
    tk.Label(parent, text=text,
             bg='#F0F4FF', fg=color,
             font=(FNT, 8, 'bold'),
             padx=8, pady=2, relief='flat').pack(side='left', padx=(0, 6))


def _chart_title(parent, text: str):
    tk.Label(parent, text=text.upper(),
             bg=BG, fg=MUTED,
             font=(FNT, 7),
             anchor='w', padx=4).pack(fill='x', pady=(4, 2))


# ── Main window ───────────────────────────────────────────────────────────────
class MainView(tk.Tk):
    def __init__(self, on_group_select):
        super().__init__()
        self.title('Wafer Analysis  ·  Selected vs Others')
        self.configure(bg=BG)
        self.state('zoomed')

        self._on_select    = on_group_select
        self._navbar:      NavBar | None         = None
        self._date_filter: DateFilterBar | None  = None
        self._page:        tk.Widget | None      = None

    def build_navbar(self, groups: list):
        self._navbar = NavBar(self, groups, self._on_select)
        self._navbar.pack(side='top', fill='x')

    def build_date_filter(self, bounds: dict, on_change):
        if not bounds:
            return
        self._date_filter = DateFilterBar(self, bounds, on_change)
        self._date_filter.pack(side='top', fill='x')

    def get_date_ranges(self) -> dict:
        if self._date_filter:
            return self._date_filter.get_ranges()
        return {}

    def render_group(self, group: str, df: pd.DataFrame, raw: dict,
                     limits: dict = None, unit: str = ''):
        if self._page:
            self._page.destroy()
        self._page = GroupPage(self, group, df, raw, limits=limits, unit=unit)
        self._page.pack(fill='both', expand=True)
        if self._navbar:
            self._navbar.set_active(group)

    def render_stat_tests(self, tab_name: str, df: pd.DataFrame):
        if self._page:
            self._page.destroy()
        self._page = StatTestPage(self, df)
        self._page.pack(fill='both', expand=True)
        if self._navbar:
            self._navbar.set_active(tab_name)
