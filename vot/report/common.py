"""Common functions for document generation."""
import os
import math
from typing import List

from vot.tracker import Tracker
from vot.report import ScatterPlot, LinePlot, Table
from vot.analysis import Measure, Point, Plot, Curve, Sorting, Axes

def read_resource(name):
    """Reads a resource file from the package directory. The file is read as a string."""
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, "r") as filehandle:
        return filehandle.read()

def per_tracker(a):
    """Returns true if the analysis is per-tracker."""
    return a.axes == Axes.TRACKERS

def extract_measures_table(trackers: List[Tracker], results) -> Table:
    """Extracts a table of measures from the results. The table is a list of lists, where each list is a column. 
    The first column is the tracker name, the second column is the measure name, and the rest of the columns are the values for each tracker.
    
    Args:
        trackers (list): List of trackers.
        results (dict): Dictionary of results. It is a dictionary of dictionaries, where the first key is the experiment, and the second key is the analysis. The value is a list of results for each tracker.
    """
    table_header = [[], [], []]
    table_data = dict()
    column_order = []

    def safe(value, default):
        return value if not value is None else default

    for experiment, eresults in results.items():
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()

            # Ignore all non per-tracker results
            if not per_tracker(analysis):
                continue

            for i, description in enumerate(descriptions):
                if description is None:
                    continue
                if isinstance(description, Measure):
                    table_header[0].append(experiment)
                    table_header[1].append(analysis)
                    table_header[2].append(description)
                    column_order.append(description.direction)

            if aresults is None:
                continue

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
            values = sorted(values, key=lambda x: safe(x[0], -math.inf), reverse=False)
        elif order == Sorting.DESCENDING:
            values = sorted(values, key=lambda x: safe(x[0], math.inf), reverse=True)
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
 
    return Table(table_header, table_data, table_order)

def extract_plots(trackers: List[Tracker], results, order=None):
    """Extracts a list of plots from the results. The list is a list of tuples, where each tuple is a pair of strings and a plot.
    
    Args:
        trackers (list): List of trackers.
        results (dict): Dictionary of results. It is a dictionary of dictionaries, where the first key is the experiment, and the second key is the analysis. The value is a list of results for each tracker.
        
    Returns:
        list: List of plots.
    """
    plots = dict()
    j = 0

    for experiment, eresults in results.items():
        experiment_plots = list()
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()

            # Ignore all non per-tracker results
            if not per_tracker(analysis):
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

                for t in order if order is not None else range(len(trackers)):
                    tracker = trackers[t]
                    values = aresults[t, 0]
                    data = values[i] if not values is None else None
                    plot(tracker, data)

                experiment_plots.append((analysis.title + " - " + description.name, plot))

        plots[experiment] = experiment_plots

    return plots

def format_value(data):
    """Formats a value for display. If the value is a string, it is returned as is. If the value is an integer, it is returned as a string. 
    If the value is a float, it is returned as a string with 3 decimal places. Otherwise, the value is converted to a string.

    Args:
        data: Value to format.

    Returns:
        str: Formatted value.
    
    """
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
    """Merges repeated objects in a list into a list of tuples (object, count)."""
    
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
