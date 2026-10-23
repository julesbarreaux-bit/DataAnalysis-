from model import WaferModel
from view  import MainView

STATS_TAB = 'Statistical Tests'


class AppController:
    def __init__(self):
        self._model = WaferModel()
        self._view  = MainView(on_group_select=self.on_group_select)
        self._current_group: str | None = None

        self._view.build_navbar(self._model.groups() + [STATS_TAB])
        bounds = self._model.date_bounds()
        self._view.build_date_filter(bounds, self.on_date_change)

    def on_group_select(self, group: str):
        self._current_group = group
        if group == STATS_TAB:
            self._view.render_stat_tests(STATS_TAB, self._model.compare_tests())
            return
        df   = self._model.stats(group)
        raw  = self._model.raw(group)
        lims = self._model.limits(group)
        unit = self._model.unit(group)
        self._view.render_group(group, df, raw, lims, unit=unit)

    def on_date_change(self):
        if not self._current_group:
            return
        self._model.set_date_filter(self._view.get_date_ranges())
        self.on_group_select(self._current_group)

    def run(self):
        groups = self._model.groups()
        if groups:
            self._view.after(80, lambda: self.on_group_select(groups[0]))
        self._view.mainloop()
