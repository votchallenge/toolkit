
import os
import typing
from abc import ABC, abstractmethod
import json
import math
import inspect
import threading
import logging
import tempfile
import datetime
import collections
from asyncio import wait
from asyncio.futures import wrap_future

import yaml

from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as colors

from vot import __version__ as version
from vot import check_debug
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment, analysis_resolver
from vot.utilities import class_fullname
from vot.utilities.attributes import Attributee, Object, Nested, String, Callable, Integer, List

class Plot(object):

    def __init__(self, identifier: str, xlabel: str, ylabel: str,
        xlimits: typing.Tuple[float, float], ylimits: typing.Tuple[float, float], trait = None):

        self._identifier = identifier

        self._manager = StyleManager.default()

        self._figure, self._axes = self._manager.make_figure(trait)

        self._axes.xaxis.set_label_text(xlabel)
        self._axes.yaxis.set_label_text(ylabel)

        if not xlimits is None and not any([x is None for x in xlimits]):
            self._axes.set_xlim(xlimits)
            self._axes.autoscale(False, axis="x")
        if not ylimits is None and not any([y is None for y in ylimits]):
            self._axes.set_ylim(ylimits)
            self._axes.autoscale(False, axis="y")

    def __call__(self, key, data):
        self.draw(key, data)

    def draw(self, key, data):
        raise NotImplementedError
    
    @property
    def axes(self) -> Axes:
        return self._axes

    def save(self, output, fmt):
        self._figure.savefig(output, format=fmt, bbox_inches='tight', transparent=True)

    @property
    def identifier(self):
        return self._identifier

class ScatterPlot(Plot):

    def draw(self, key, data):
        if data is None or len(data) != 2:
            return

        style = self._manager.plot_style(key)
        handle = self._axes.scatter(data[0], data[1], **style.point_style())
        #handle.set_gid("report_%s_%d" % (self._identifier, style["number"]))

class LinePlot(Plot):

    def draw(self, key, data):
        if data is None or len(data) < 1:
            return

        if isinstance(data[0], tuple):
            # Drawing curve
            if len(data[0]) != 2:
                return
            x, y = zip(*data)
        else:
            y = data
            x = range(len(data))

        style = self._manager.plot_style(key)

        handle = self._axes.plot(x, y, **style.line_style())
       # handle[0].set_gid("report_%s_%d" % (self._identifier, style["number"]))

def generate_serialized(trackers: typing.List[Tracker], sequences: typing.List[Sequence], results, storage: "Storage", serializer: str):

    doc = dict()
    doc["toolkit"] = version
    doc["timestamp"] = datetime.datetime.now().isoformat()
    doc["trackers"] = {t.reference : t.describe() for t in trackers}
    doc["sequences"] = {s.name : s.describe() for s in sequences}

    doc["results"] = dict()

    for experiment, analyses in results.items():
        exp = dict(parameters=experiment.dump(), type=class_fullname(experiment))
        exp["results"] = []
        for _, data in analyses.items():
            exp["results"].append(data)
        doc["results"][experiment.identifier] = exp

    if serializer == "json":
        with storage.write("results.json") as handle:
            json.dump(doc, handle, indent=2)
    elif serializer == "yaml":
        with storage.write("results.yaml") as handle:
            yaml.dump(doc, handle)
    else:
        raise RuntimeError("Unknown serializer")

def configure_axes(figure, rect=None, _=None):

    axes = Axes(figure, rect or [0, 0, 1, 1])

    figure.add_axes(axes)

    return axes

def configure_figure(traits=None):

    args = {}
    if traits == "ar":
        args["figsize"] = (5, 5)
    elif traits == "eao":
        args["figsize"] = (7, 5)
    elif traits == "attributes":
        args["figsize"] = (15, 5)

    return Figure(**args)

class PlotStyle(object):

    def line_style(self, opacity=1):
        raise NotImplementedError

    def point_style(self):
        raise NotImplementedError

class DefaultStyle(PlotStyle):

    colormap = get_cmap("tab20b")
    colorcount = 20
    markers = ["o", "v", "<", ">", "^", "8", "*"]

    def __init__(self, number):
        super().__init__()
        self._number = number

    def line_style(self, opacity=1):
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        if opacity < 1:
            color = colors.to_rgba(color, opacity)
        return dict(linewidth=1, c=color)

    def point_style(self):
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        marker = DefaultStyle.markers[self._number % len(DefaultStyle.markers)]
        return dict(marker=marker, c=[color])

class Legend(object):

    def __init__(self, style_factory=DefaultStyle):
        self._mapping = collections.OrderedDict()
        self._counter = 0
        self._style_factory = style_factory

    def _number(self, key):
        if not key in self._mapping:
            self._mapping[key] = self._counter
            self._counter += 1
        return self._mapping[key]

    def __getitem__(self, key) -> PlotStyle:
        number = self._number(key)
        return self._style_factory(number)

    def _style(self, number):
        raise NotImplementedError

    def keys(self):
        return self._mapping.keys()

    def figure(self, key):
        style = self[key]
        figure = Figure(figsize=(0.1, 0.1))  # TODO: hardcoded
        axes = Axes(figure, [0, 0, 1, 1], yticks=[], xticks=[], frame_on=False)
        figure.add_axes(axes)
        axes.patch.set_visible(False)
        marker_style = style.point_style()
        marker_style["s"] = 40 # Reset size
        axes.scatter(0, 0, **marker_style)
        return figure

class StyleManager(Attributee):

    plots = Callable(default=DefaultStyle)
    axes = Callable(default=configure_axes)
    figure = Callable(default=configure_figure)

    _context = threading.local()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._legends = dict()

    def __getitem__(self, key) -> PlotStyle:
        return self.plot_style(key)

    def legend(self, key) -> Legend:
        if inspect.isclass(key):
            klass = key
        else:
            klass = type(key)

        if not klass in self._legends:
            self._legends[klass] = Legend(self.plots)

        return self._legends[klass]

    def plot_style(self, key) -> PlotStyle:
        return self.legend(key)[key]

    def make_axes(self, figure, rect=None, trait=None) -> Axes:
        return self.axes(figure, rect, trait)

    def make_figure(self, trait=None) -> typing.Tuple[Figure, Axes]:
        figure = self.figure(trait)
        axes = self.make_axes(figure, trait=trait)

        return figure, axes

    def __enter__(self):

        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            return self

        StyleManager._context.style_manager = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            StyleManager._context.style_manager = None

    @staticmethod
    def default() -> "StyleManager":

        manager = getattr(StyleManager._context, 'style_manager', None)
        if manager is None:
            manager = StyleManager()
            StyleManager._context.style_manager = manager

        return manager

class TrackerSorter(Attributee):

    experiment = String(default=None)
    analysis = String(default=None)
    result = Integer(val_min=0, default=0)

    def __call__(self, experiments, trackers, sequences):
        if self.experiment is None or self.analysis is None:
            return range(len(trackers))

        experiment = next(filter(lambda x: x.identifier == self.experiment, experiments), None)

        if experiment is None:
            raise RuntimeError("Experiment not found")

        analysis = next(filter(lambda x: x.name == self.analysis, experiment.analyses), None)

        if analysis is None:
            raise RuntimeError("Analysis not found")

        future = analysis.commit(experiment, trackers, sequences)
        result = future.result()

        scores = [x[self.result] for x in result]
        indices = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])]

        return indices

class Generator(Attributee):

    async def generate(self, experiments, trackers, sequences):
        raise NotImplementedError

    async def process(self, analyses, experiment, trackers, sequences):
        if not isinstance(analyses, collections.Iterable):
            analyses = [analyses]

        futures = []

        for analysis in analyses:
            futures.append(wrap_future(analysis.commit(experiment, trackers, sequences)))

        await wait(futures)

        if len(futures) == 1:
            return futures[0].result()
        else:
            return (future.result() for future in futures)

class ReportConfiguration(Attributee):

    style = Nested(StyleManager)
    sort = Nested(TrackerSorter)
    generators = List(Object(subclass=Generator), default=[])

# TODO: replace this with report generator and separate json/yaml dump
def generate_document(format: str, config: ReportConfiguration, trackers: typing.List[Tracker], sequences: typing.List[Sequence], results, storage: "Storage"):

    from .html import generate_html_document
    from .latex import generate_latex_document
    from .common import wrt_trackers

    if format == "json":
        generate_serialized(trackers, sequences, results, storage, "json")
    elif format == "yaml":
        generate_serialized(trackers, sequences, results, storage, "yaml")
    else:
        order = config.sort(results.keys(), trackers, sequences)

        trackers = [trackers[i] for i in order]

        for _, eresults in results.items():
            for analysis, aresults in eresults.items():
                if aresults is None:
                    eresults[analysis] = [None] * len(order)
                    continue
                if wrt_trackers(analysis.axes()) is None:
                    continue
                eresults[analysis] = [aresults[i] for i in order]

        with config.style:
            if format == "html":
                generate_html_document(trackers, sequences, results, storage)
            elif format == "latex":
                generate_latex_document(trackers, sequences, results, storage, False)
            elif format == "pdf":
                generate_latex_document(trackers, sequences, results, storage, True)

