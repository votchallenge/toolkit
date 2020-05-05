
from typing import List, Any, Tuple
from abc import ABC, abstractmethod
import json
import logging
import tempfile
import collections

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from pylatex import Document, Section, Subsection, Command, LongTable, MultiColumn
from pylatex import Figure as TeXFigure
from pylatex.utils import italic, NoEscape

from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.workspace import Storage
from vot.analysis import Analysis, Measure, Point, Plot, Curve

def _extract_measures_table(results):
    table_header = [[], [], []]
    table_data = dict()

    for experiment, eresults in results.items():
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()
            
            for i, description in enumerate(descriptions):
                if description is None:
                    continue
                if isinstance(description, Measure):
                    table_header[0].append(experiment)
                    table_header[1].append(analysis)
                    table_header[2].append(description)

            for tracker, values in aresults.items():
                if not tracker in table_data:
                    table_data[tracker] = list()
                for i, description in enumerate(descriptions):
                    if description is None:
                        continue
                    if isinstance(description, Measure):
                        table_data[tracker].append(values[i] if not values is None else None)

    return table_header, table_data

class Legend(object):

    def __init__(self):
        self._mapping = collections.OrderedDict()
        self._counter = 0

    def __getitem__(self, key):

        if not key in self._mapping:
            self._mapping[key] = self._counter
            self._counter += 1
        return self._mapping[key]

class Figure(ABC):

    def __init__(self, legend: Legend, xlimits: Tuple[float, float], ylimits: Tuple[float, float]):
        self._handle = plt.figure()
        self._legend = legend if not legend is None else Legend()
        with self:
            if not xlimits is None:
                plt.xlim(xlimits)
            if not ylimits is None:
                plt.ylim(ylimits)

    def __enter__(self):
        plt.figure(self._handle.number)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __call__(self, name, data):
        self.draw(name, data)

    @abstractmethod
    def draw(self, name, data):
        raise NotImplementedError

    def _unique(self, name):
        return self._legend[name]

class DefaultStyleFigure(Figure, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._colormap = get_cmap("tab20b")
        self._colorcount = 20
        self._markers = ["o", "v", "<", ">", "^", "8", "*"]
    
    def style(self, name):
        number = self._unique(name)
        color = self._colormap((number % self._colorcount + 1) / self._colorcount)
        marker = self._markers[number % len(self._markers)]
        return dict(color=color, marker=marker, number=number+1, width=1, style=None)

class ScatterPlot(DefaultStyleFigure):

    def draw(self, name, data):
        if len(data) != 2:
            return

        style = self.style(name)

        with self:
            plt.scatter(data[0], data[1], marker=style["marker"], c=[style["color"]])


class LinePlot(DefaultStyleFigure):

    def draw(self, name, data):
        if len(data) < 1:
            return

        if isinstance(data[0], tuple):
            # Drawing curve
            if len(data[0]) != 2:
                return
            x, y = zip(*data)
        else:
            y = data
            x = range(len(data))

        style = self.style(name)

        with self:
            plt.plot(x, y, c=[style["color"]], linewidth=style["width"])


def _extract_graphs(results):
    graphs = dict()
    legend = Legend()

    for experiment, eresults in results.items():
        experiment_graphs = list()
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()

            for i, description in enumerate(descriptions):
                if description is None:
                    continue
                if isinstance(description, Point) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    graph = ScatterPlot(legend, xlim, ylim)
                elif isinstance(description, Plot):
                    ylim = (description.minimal(), description.maximal())
                    graph = LinePlot(legend, None, ylim)
                elif isinstance(description, Curve) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    graph = LinePlot(legend, xlim, ylim)
                else:
                    continue

                for tracker, values in aresults.items():
                    data = values[i] if not values is None else None
                    graph(tracker.label, data)

                experiment_graphs.append((description.name, graph))

        graphs[experiment] = experiment_graphs

    return graphs

def _merge_repeats(objects):
    
    if not objects:
        return []

    repeats = []
    previous = objects[0]
    count = 1

    for o in objects[1:]:
        if o == previous:
            count = count + 1
        else:
            repeats.append((previous, count))
            previous = o
            count = 1

    repeats.append((previous, count))

    return repeats

def generate_json_document(results, storage: Storage):

    def transform_key(key):
        if isinstance(key, Analysis):
            return key.name
        if isinstance(key, Tracker):
            return key.label
        if isinstance(key, Experiment):
            return key.identifier
        return key

    def transform_value(value):
        if isinstance(value, dict):
            return {transform_key(k): transform_value(v) for k, v in value.items()}
        return value

    with storage.write("results.json") as handle:
        json.dump(transform_value(results), handle, indent=2)


def generate_latex_document(results, storage: Storage, build=False):

    logger = logging.getLogger("vot")

    doc = Document(page_numbers=True)

    doc.preamble.append(Command('title', 'VOT toolkit report'))
    doc.preamble.append(Command('author', 'Anonymous author'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    table_header, table_data = _extract_measures_table(results)

    if len(table_header[2]) == 0:
        logger.debug("No measures found, skipping table")
    else:
        # Generate data table
        with doc.create(LongTable("l " * (len(table_header[2]) + 1))) as data_table:
            data_table.add_hline()
            data_table.add_row([" "] + [MultiColumn(c[1], data=c[0].identifier) for c in _merge_repeats(table_header[0])])
            data_table.add_hline()
            data_table.add_row([" "] + [MultiColumn(c[1], data=c[0].name) for c in _merge_repeats(table_header[1])])
            data_table.add_hline()
            data_table.add_row([" "] + [c.abbreviation for c in table_header[2] ])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()

            for tracker, data in table_data.items():
                data_table.add_row([tracker.label] + data)

    graphs = _extract_graphs(results)

    for experiment, experiment_graphs in graphs.items():
        if len(experiment_graphs) == 0:
            continue

        doc.append(Section("Experiment " + experiment.identifier))

        for title, graph in experiment_graphs:

            with graph, doc.create(TeXFigure(position='htbp')) as plot:
                plot.add_plot()
                plot.add_caption(title)
                

    temp = tempfile.mktemp()
    logger.debug("Generating to tempourary output %s", temp)

    if build:
        doc.generate_pdf(temp, clean_tex=True)
        storage.copy(temp + ".pdf", "report.pdf")
    else:
        doc.generate_tex(temp)
        storage.copy(temp + ".tex", "report.tex")

def generate_html_document(results, storage: Storage):
    raise NotImplementedError