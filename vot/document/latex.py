
import io
import logging
import tempfile
import datetime
from typing import List

from pylatex.base_classes import Container
from pylatex.package import Package
from pylatex import Document, Section, Command, LongTable, MultiColumn, MultiRow, Figure, UnsafeCommand
from pylatex.utils import NoEscape

from vot import toolkit_version
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.workspace import Storage
from vot.document.common import format_value, read_resource, merge_repeats, extract_measures_table, extract_plots
from vot.document import StyleManager

TRACKER_GROUP = "default"

class Chunk(Container):

    def dumps(self):
        return self.dumps_content()

def strip_comments(src, wrapper=True):
    return "\n".join([line for line in src.split("\n") if not line.startswith("%") and (wrapper or not line.startswith(r"\makeat"))])

def insert_figure(figure):
    buffer = io.StringIO()
    figure.save(buffer, "PGF")
    return NoEscape(strip_comments(buffer.getvalue()))

def insert_mplfigure(figure, wrapper=True):
    buffer = io.StringIO()
    figure.savefig(buffer, format="PGF", bbox_inches='tight', pad_inches=0.01)
    return NoEscape(strip_comments(buffer.getvalue(), wrapper))


def generate_symbols(container, trackers):

    legend = StyleManager.default().legend(Tracker)

    container.append(Command("makeatletter"))
    for tracker in trackers:
        container.append(UnsafeCommand('DefineTracker', [tracker.reference, TRACKER_GROUP],
             extra_arguments=insert_mplfigure(legend.figure(tracker), False) + r' \replunderscores{%s}' % tracker.label))

    container.append(Command("makeatother"))


def generate_latex_document(trackers: List[Tracker], sequences: List[Sequence], results, storage: Storage, build=False, multipart=True):

    order_marks = {1: "first", 2: "second", 3: "third"}

    def format_cell(value, order):
        cell = format_value(value)
        if order in order_marks:
            cell = Command(order_marks[order], cell)
        return cell

    logger = logging.getLogger("vot")

    table_header, table_data, table_order = extract_measures_table(trackers, results)
    plots = extract_plots(trackers, results)

    doc = Document(page_numbers=True)

    doc.preamble.append(Package('pgf'))
    doc.preamble.append(Package('xcolor'))
    doc.preamble.append(Package('fullpage'))

    doc.preamble.append(NoEscape(read_resource("commands.tex")))

    doc.preamble.append(UnsafeCommand('newcommand', r'\first', options=1, extra_arguments=r'{\color{red} #1 }'))
    doc.preamble.append(UnsafeCommand('newcommand', r'\second', options=1, extra_arguments=r'{\color{green} #1 }'))
    doc.preamble.append(UnsafeCommand('newcommand', r'\third', options=1, extra_arguments=r'{\color{blue} #1 }'))

    if multipart:
        container = Chunk()
        generate_symbols(container, trackers)
        with storage.write("symbols.tex") as out:
            container.dump(out)
        doc.preamble.append(Command("input", "symbols.tex"))
    else:
        generate_symbols(doc.preamble, trackers)

    doc.preamble.append(Command('title', 'VOT report'))
    doc.preamble.append(Command('author', 'Toolkit version ' + toolkit_version()))
    doc.preamble.append(Command('date', datetime.datetime.now().isoformat()))
    doc.append(NoEscape(r'\maketitle'))

    if len(table_header[2]) == 0:
        logger.debug("No measures found, skipping table")
    else:

        # Generate data table
        with doc.create(LongTable("l " * (len(table_header[2]) + 1))) as data_table:
            data_table.add_hline()
            data_table.add_row([" "] + [MultiColumn(c[1], data=c[0].identifier) for c in merge_repeats(table_header[0])])
            data_table.add_hline()
            data_table.add_row([" "] + [MultiColumn(c[1], data=c[0].title) for c in merge_repeats(table_header[1])])
            data_table.add_hline()
            data_table.add_row(["Tracker"] + [" " + c.abbreviation + " " for c in table_header[2]])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()

            for tracker, data in table_data.items():
                data_table.add_row([UnsafeCommand("Tracker", [tracker.reference, TRACKER_GROUP])] +
                    [format_cell(x, order[tracker] if not order is None else None) for x, order in zip(data, table_order)])

    for experiment, experiment_plots in plots.items():
        if len(experiment_plots) == 0:
            continue

        doc.append(Section("Experiment " + experiment.identifier))

        for title, plot in experiment_plots:

            with doc.create(Figure(position='htbp')) as container:
                if multipart:
                    plot_name = plot.identifier + ".pdf"
                    with storage.write(plot_name, binary=True) as out:
                        plot.save(out, "PDF")
                    container.add_image(plot_name)
                else:
                    container.append(insert_figure(plot))
                container.add_caption(title)
                
    if build:
        temp = tempfile.mktemp()
        logger.debug("Generating to tempourary output %s", temp)
        doc.generate_pdf(temp, clean_tex=True)
        storage.copy(temp + ".pdf", "report.pdf")
    else:
        with storage.write("report.tex") as out:
            doc.dump(out)
