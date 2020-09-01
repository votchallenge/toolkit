
import os
import io
import logging
import datetime
from typing import List

import dominate
from dominate.tags import h1, h2, table, thead, tbody, tr, th, td, div, p, li, ol, span, style, link, script
from dominate.util import raw

from vot import toolkit_version, check_debug
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.workspace import Storage
from vot.document.common import format_value, read_resource, merge_repeats, extract_measures_table, extract_plots
from vot.document import StyleManager
from vot.utilities.data import Grid

ORDER_CLASSES = {1: "first", 2: "second", 3: "third"}

def insert_cell(value, order):
    attrs = dict(data_sort_value=order, data_value=value)
    if order in ORDER_CLASSES:
        attrs["cls"] = ORDER_CLASSES[order]
    td(format_value(value), **attrs)

def grid_table(data: Grid, rows: List[str], columns: List[str]):

    assert data.dimensions == 2
    assert data.size(0) == len(rows) and data.size(1) == len(columns)

    with table() as element:
        with thead():
            with tr():
                th()
                [th(column) for column in columns]
        with tbody():
            for i, row in enumerate(rows):
                with tr():
                    th(row)
                    for value in data.row(i):
                        if isinstance(value, tuple):
                            if len(value) == 1:
                                value = value[0]
                        insert_cell(value, None)

    return element

def generate_html_document(trackers: List[Tracker], sequences: List[Sequence], results, storage: Storage):

    def insert_figure(figure):
        buffer = io.StringIO()
        figure.save(buffer, "SVG")
        raw(buffer.getvalue())

    def insert_mplfigure(figure):
        buffer = io.StringIO()
        figure.savefig(buffer, format="SVG", bbox_inches='tight', pad_inches=0.01, dpi=200)
        raw(buffer.getvalue())

    def add_style(name, linked=False):
        if linked:
            link(rel='stylesheet', href='file://' + os.path.join(os.path.dirname(__file__), name))
        else:
            style(read_resource(name))

    def add_script(name, linked=False):
        if linked:
            script(type='text/javascript', src='file://' + os.path.join(os.path.dirname(__file__), name))
        else:
            with script(type='text/javascript'):
                raw("//<![CDATA[\n" + read_resource(name) + "\n//]]>")

    logger = logging.getLogger("vot")

    table_header, table_data, table_order = extract_measures_table(trackers, results)
    plots = extract_plots(trackers, results)

    legend = StyleManager.default().legend(Tracker)

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
            li('Toolkit version: ' + toolkit_version())
            li('Created: ' + datetime.datetime.now().isoformat())

        if len(table_header[2]) == 0:
            logger.debug("No measures found, skipping table")
        else:
            with table(cls="overview-table pure-table pure-table-horizontal pure-table-striped"):
                with thead():
                    with tr():
                        th()
                        [th(c[0].identifier, colspan=c[1]) for c in merge_repeats(table_header[0])]
                    with tr():
                        th()
                        [th(c[0].title, colspan=c[1]) for c in merge_repeats(table_header[1])]
                    with tr():
                        th("Trackers")
                        [th(c.abbreviation, data_sort="int" if order else "") for c, order in zip(table_header[2], table_order)]
                with tbody():
                    for tracker, data in table_data.items():
                        with tr(data_tracker=tracker.reference):
                            with td():
                                insert_mplfigure(legend.figure(tracker))
                                span(tracker.label)
                            for value, order in zip(data, table_order):
                                insert_cell(value, order[tracker] if not order is None else None)

        for experiment, experiment_plots in plots.items():
            if len(experiment_plots) == 0:
                continue

            h2("Experiment {}".format(experiment.identifier), cls="experiment")

            with div(cls="plots"):

                for title, plot in experiment_plots:
                    with div(cls="plot"):
                        p(title)
                        insert_figure(plot)
                        

    with storage.write("report.html") as filehandle:
        filehandle.write(doc.render())
