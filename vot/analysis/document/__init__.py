
import os
from typing import List, Any, Tuple
from abc import ABC, abstractmethod
import json
import math
import logging
import tempfile
import datetime
import collections

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from vot import __version__ as version
from vot import check_debug
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.workspace import Storage
from vot.analysis import Analysis, Measure, Point, Plot, Curve, Hints, Sorting, Axis

def _extract_measures_table(trackers, results):
    table_header = [[], [], []]
    table_data = dict()
    column_order = []

    for experiment, eresults in results.items():
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()
            axes = analysis.axes()

            # Ignore all non per-tracker results
            if axes is None or len(axes) != 1 or axes[0] != Axis.TRACKERS:
                continue

            for i, description in enumerate(descriptions):
                if description is None:
                    continue
                if isinstance(description, Measure):
                    table_header[0].append(experiment)
                    table_header[1].append(analysis)
                    table_header[2].append(description)
                    column_order.append(description.direction)

            for tracker, values in zip(trackers, aresults):
                if not tracker in table_data:
                    table_data[tracker] = list()
                for i, description in enumerate(descriptions):
                    if description is None:
                        continue
                    if isinstance(description, Measure):
                        table_data[tracker].append(values[i] if not values is None else None)

    table_order = []

    for i, order in enumerate(column_order):
        values = [(v[i], k) for k, v in table_data.items()]
        if order == Sorting.ASCENDING:
            values = sorted(values, key=lambda x: x[0] or -math.inf, reverse=False)
        elif order == Sorting.DESCENDING:
            values = sorted(values, key=lambda x: x[0] or math.inf, reverse=True)
        else:
            table_order.append(None)
            continue
        order = dict()
        j = 0
        value = None
        # Take into account that some values are the same
        for k, v in enumerate(values):
            j = j if value == v[0] else k + 1
            value = v[0]
            order[v[1]] = j
        table_order.append(order)

    return table_header, table_data, table_order

class Legend(object):

    def __init__(self):
        self._mapping = collections.OrderedDict()
        self._counter = 0
        self._colormap = get_cmap("tab20b")
        self._colorcount = 20
        self._markers = ["o", "v", "<", ">", "^", "8", "*"]

    def number(self, name):
        if not name in self._mapping:
            self._mapping[name] = self._counter
            self._counter += 1
        return self._mapping[name]

    def __getitem__(self, name):
        number = self.number(name)
        color = self._colormap((number % self._colorcount + 1) / self._colorcount)
        marker = self._markers[number % len(self._markers)]
        return dict(color=color, marker=marker, number=number, width=1, name=name)

    def names(self):
        return self._mapping.keys()

    def figure(self, name):
        from matplotlib.axes import Axes
        style = self[name]
        plt.figure(figsize=(0.1, 0.1)) # TODO: hardcoded
        ax = Axes(plt.gcf(), [0, 0, 1, 1], yticks=[], xticks=[], frame_on=False)
        plt.gcf().delaxes(plt.gca())
        plt.gcf().add_axes(ax)
        #plt.axis('off')
        plt.scatter(0, 0, marker=style["marker"], c=[style["color"]])
        plt.autoscale(tight=True)


class Figure(ABC):

    def __init__(self, identifier: str, legend: Legend, xlabel: str, ylabel: str,
        xlimits: Tuple[float, float], ylimits: Tuple[float, float], hints: Hints):
        figargs = dict()
        if hints & Hints.AXIS_EQUAL:
            figargs["figsize"] = (5, 5)

        self._handle = plt.figure(**figargs)
        self._legend = legend if not legend is None else Legend()
        self._identifier = identifier

        if hints & Hints.AXIS_EQUAL:
            plt.axis('equal')

        with self:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if not xlimits is None:
                plt.xlim(xlimits)
                plt.autoscale(False, axis="x")
            if not ylimits is None:
                plt.ylim(ylimits)
                plt.autoscale(False, axis="y")



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

    def style(self, name):
        return self._legend[name]

class ScatterPlot(Figure):

    def draw(self, name, data):
        if data is None or len(data) != 2:
            return

        style = self.style(name)

        with self:
            handle = plt.scatter(data[0], data[1], marker=style["marker"], c=[style["color"]])
            handle.set_gid("report_%s_%d" % (self._identifier, style["number"]))


class LinePlot(Figure):

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
            handle = plt.plot(x, y, c=style["color"], linewidth=style["width"])
            handle[0].set_gid("report_%s_%d" % (self._identifier, style["number"]))


def _extract_graphs(trackers, results, legend):
    graphs = dict()
    j = 0

    for experiment, eresults in results.items():
        experiment_graphs = list()
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()
            axes = analysis.axes()

            # Ignore all non per-tracker results
            if axes is None or len(axes) != 1 or axes[0] != Axis.TRACKERS:
                continue

            for i, description in enumerate(descriptions):
                if description is None:
                    continue

                graph_identifier = "%s_%d" % (experiment.identifier, j)
                j += 1

                if isinstance(description, Point) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    xlabel = description.label(0)
                    ylabel = description.label(1)
                    graph = ScatterPlot(graph_identifier, legend, xlabel, ylabel, xlim, ylim, description.hints)
                elif isinstance(description, Plot):
                    ylim = (description.minimal, description.maximal)
                    graph = LinePlot(graph_identifier, legend, description.name, description.wrt, None, ylim, description.hints)
                elif isinstance(description, Curve) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    xlabel = description.label(0)
                    ylabel = description.label(1)
                    graph = LinePlot(graph_identifier, legend, xlabel, ylabel, xlim, ylim, description.hints)
                else:
                    continue

                for tracker, values in zip(trackers, aresults):
                    data = values[i] if not values is None else None
                    graph(tracker.identifier, data)

                experiment_graphs.append((description.name, graph))

        graphs[experiment] = experiment_graphs

    return graphs

def _format_value(data):
    if data is None:
        return "N/A"
    if isinstance(data, str):
        return data
    if isinstance(data, int):
        return "%d" % data
    if isinstance(data, float):
        return "%.4f" % data
    return str(data)

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

def _read_resource(name):
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, "r") as filehandle:
        return filehandle.read()

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


def generate_latex_document(trackers: List[Tracker], sequences: List[Sequence], results, storage: Storage, build=False):

    from pylatex import Document, Section, Command, LongTable, MultiColumn, Figure
    from pylatex.utils import NoEscape

    order_marks = {1: "first", 2: "second", 3: "third"}

    def format_cell(value, order):
        cell = _format_value(value)
        if order in order_marks:
            cell = Command(order_marks[order], cell)
        return cell

    logger = logging.getLogger("vot")

    legend = Legend()
    table_header, table_data, table_order = _extract_measures_table(trackers, results)
    graphs = _extract_graphs(trackers, results, legend)

    doc = Document(page_numbers=True)

    doc.preamble.append(Command('title', 'VOT report'))
    doc.preamble.append(Command('author', 'Toolkit version ' + version))
    doc.preamble.append(Command('date', datetime.datetime.now().isoformat()))
    doc.append(NoEscape(r'\maketitle'))


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
            data_table.add_row(["Trackers"] + [c.abbreviation for c in table_header[2]])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()

            for tracker, data in table_data.items():
                data_table.add_row([tracker.label] +
                    [format_cell(x, order[tracker] if not order is None else None) for x, order in enumerate(data, table_order)])

    for experiment, experiment_graphs in graphs.items():
        if len(experiment_graphs) == 0:
            continue

        doc.append(Section("Experiment " + experiment.identifier))

        for title, graph in experiment_graphs:

            with graph, doc.create(Figure(position='htbp')) as plot:
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

def generate_html_document(trackers: List[Tracker], sequences: List[Sequence], results, storage: Storage):

    import io
    import dominate
    from dominate.tags import h1, h2, table, thead, tbody, tr, th, td, div, p, li, ol, span, style, link, script
    from dominate.util import raw

    order_classes = {1: "first", 2: "second", 3: "third"}

    def insert_figure():
        buffer = io.StringIO()
        plt.savefig(buffer, format="SVG", bbox_inches='tight', transparent=True)
        raw(buffer.getvalue())
        plt.close()

    def insert_cell(value, order):
        attrs = dict(data_sort_value=order, data_value=value)
        if order in order_classes:
            attrs["cls"] = order_classes[order]
        td(_format_value(value), **attrs)

    def add_style(name, linked=False):
        if linked:
            link(rel='stylesheet', href='file://' + os.path.join(os.path.dirname(__file__), name))
        else:
            style(_read_resource(name))

    def add_script(name, linked=False):
        if linked:
            script(type='text/javascript', src='file://' + os.path.join(os.path.dirname(__file__), name))
        else:
            with script(type='text/javascript'):
                raw("//<![CDATA[\n" + _read_resource(name) + "\n//]]>")

    logger = logging.getLogger("vot")

    legend = Legend()
    table_header, table_data, table_order = _extract_measures_table(trackers, results)
    graphs = _extract_graphs(trackers ,results, legend)

    doc = dominate.document(title='VOT report')

    linked = check_debug()

    with doc.head:
        add_style("pure.css", linked)
        add_style("report.css", linked)
        add_script("jquery.js", linked)
        add_script("table.js", linked)
        add_script("report.js", linked)

    with doc:

        h1("VOT report")

        with ol(cls="metadata"):
            li('Toolkit version: ' + version)
            li('Created: ' + datetime.datetime.now().isoformat())

        if len(table_header[2]) == 0:
            logger.debug("No measures found, skipping table")
        else:
            with table(cls="measures pure-table pure-table-horizontal pure-table-striped"):
                with thead():
                    with tr():
                        th()
                        [th(c[0].identifier, colspan=c[1]) for c in _merge_repeats(table_header[0])]
                    with tr():
                        th()
                        [th(c[0].name, colspan=c[1]) for c in _merge_repeats(table_header[1])]
                    with tr():
                        th("Trackers")
                        [th(c.abbreviation, data_sort="int" if order else "") for c, order in zip(table_header[2], table_order)]
                with tbody():
                    for tracker, data in table_data.items():
                        with tr():
                            number = legend.number(tracker.identifier)
                            with td(id="legend_%d" % number):
                                legend.figure(tracker.identifier)
                                insert_figure()
                                span(tracker.label)
                            for value, order in zip(data, table_order):
                                insert_cell(value, order[tracker] if not order is None else None)

        for experiment, experiment_graphs in graphs.items():
            if len(experiment_graphs) == 0:
                continue

            h2("Experiment {}".format(experiment.identifier), cls="experiment")

            with div(cls="graphs"):

                for title, graph in experiment_graphs:
                    with graph, div(cls="graph"):
                        p(title)
                        insert_figure()
                        

    with storage.write("report.html") as filehandle:
        filehandle.write(doc.render())
