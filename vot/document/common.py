import os
import math

from vot.document import ScatterPlot, LinePlot
from vot.analysis import Measure, Point, Plot, Curve, Sorting, Axis

def read_resource(name):
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, "r") as filehandle:
        return filehandle.read()

def wrt_trackers(axes):
    if axes is None:
        return None

    return next(filter(lambda x: x == Axis.TRACKERS, axes), None)

def extract_measures_table(trackers, results):
    table_header = [[], [], []]
    table_data = dict()
    column_order = []

    for experiment, eresults in results.items():
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()

            # Ignore all non per-tracker results
            if wrt_trackers(analysis.axes()) is None:
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



def extract_plots(trackers, results):
    plots = dict()
    j = 0

    for experiment, eresults in results.items():
        experiment_plots = list()
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()
            axes = analysis.axes()

            # Ignore all non per-tracker results
            if axes is None or len(axes) != 1 or axes[0] != Axis.TRACKERS:
                continue

            for i, description in enumerate(descriptions):
                if description is None:
                    continue

                plot_identifier = "%s_%s_%d" % (experiment.identifier, analysis.name, j)
                j += 1

                if isinstance(description, Point) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    xlabel = description.label(0)
                    ylabel = description.label(1)
                    plot = ScatterPlot(plot_identifier, xlabel, ylabel, xlim, ylim, description.trait)
                elif isinstance(description, Plot):
                    ylim = (description.minimal, description.maximal)
                    plot = LinePlot(plot_identifier, description.wrt, description.name, None, ylim, description.trait)
                elif isinstance(description, Curve) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    xlabel = description.label(0)
                    ylabel = description.label(1)
                    plot = LinePlot(plot_identifier, xlabel, ylabel, xlim, ylim, description.trait)
                else:
                    continue

                for tracker, values in zip(trackers, aresults):
                    data = values[i] if not values is None else None
                    plot(tracker, data)

                experiment_plots.append((description.name, plot))

        plots[experiment] = experiment_plots

    return plots

def format_value(data):
    if data is None:
        return "N/A"
    if isinstance(data, str):
        return data
    if isinstance(data, int):
        return "%d" % data
    if isinstance(data, float):
        return "%.3f" % data
    return str(data)

def merge_repeats(objects):
    
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
