"""This module contains functions for generating LaTeX documents with results."""
import io
import tempfile
import datetime
from typing import List

from pylatex.base_classes import Container
from pylatex.package import Package
from pylatex import Document, Section, Command, LongTable, MultiColumn, Figure, UnsafeCommand
from pylatex.utils import NoEscape

from vot import toolkit_version, get_logger
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.workspace import Storage
from vot.report.common import format_value, read_resource, merge_repeats
from vot.report import StyleManager, Plot, Table

TRACKER_GROUP = "default"

class Chunk(Container):
    """A container that does not add a newline after the content."""

    def dumps(self):
        """Returns the LaTeX representation of the container."""
        return self.dumps_content()

def strip_comments(src, wrapper=True):
    """Strips comments from a LaTeX source file."""
    return "\n".join([line for line in src.split("\n") if not line.startswith("%") and (wrapper or not line.startswith(r"\makeat"))])

def insert_figure(figure):
    """Inserts a figure into a LaTeX document."""
    buffer = io.StringIO()
    figure.save(buffer, "PGF")
    return NoEscape(strip_comments(buffer.getvalue()))

def insert_mplfigure(figure, wrapper=True):
    """Inserts a matplotlib figure into a LaTeX document."""
    buffer = io.StringIO()
    figure.savefig(buffer, format="PGF", bbox_inches='tight', pad_inches=0.01)
    return NoEscape(strip_comments(buffer.getvalue(), wrapper))


def generate_symbols(container, trackers):
    """Generates a LaTeX command for each tracker. The command is named after the tracker reference and contains the tracker symbol."""

    legend = StyleManager.default().legend(Tracker)

    container.append(Command("makeatletter"))
    for tracker in trackers:
        container.append(UnsafeCommand('DefineTracker', [tracker.reference, TRACKER_GROUP],
             extra_arguments=insert_mplfigure(legend.figure(tracker), False) + r' \replunderscores{%s}' % tracker.label))

    container.append(Command("makeatother"))

def generate_latex_document(trackers: List[Tracker], sequences: List[Sequence], reports, storage: Storage, multipart=True) -> str:
    """Generates a LaTeX document with the results. The document is returned as a string. If build is True, the document is compiled and the PDF is returned.
    
    Args:
        
        trackers (list): List of trackers.
        sequences (list): List of sequences.
        reports (list): List of results tuples.
        storage (Storage): Storage object.
        multipart (bool): If True, the document is split into multiple files.
    """

    order_marks = {1: "first", 2: "second", 3: "third"}

    def format_cell(value, order):
        """Formats a cell in the data table."""
        cell = format_value(value)
        if order in order_marks:
            cell = Command(order_marks[order], cell)
        return cell

    logger = get_logger()

    doc = Document(page_numbers=True)

    doc.preamble.append(Package('pgf'))
    doc.preamble.append(Package('xcolor'))
    doc.preamble.append(Package('fullpage'))

    doc.preamble.append(NoEscape(read_resource("commands.tex")))

    doc.preamble.append(UnsafeCommand('newcommand', r'\first', options=1, extra_arguments=r'{\color{red} #1 }'))
    doc.preamble.append(UnsafeCommand('newcommand', r'\second', options=1, extra_arguments=r'{\color{green} #1 }'))
    doc.preamble.append(UnsafeCommand('newcommand', r'\third', options=1, extra_arguments=r'{\color{blue} #1 }'))

    # TODO: make table more general (now it assumes a tracker per row)
    def make_table(doc, table):

        if len(table.header[2]) == 0:
            logger.debug("No measures found, skipping table")
        else:

            # Generate data table
            with doc.create(LongTable("l " * (len(table.header[2]) + 1))) as data_table:
                data_table.add_hline()
                data_table.add_row([" "] + [MultiColumn(c[1], data=c[0].identifier) for c in merge_repeats(table.header[0])])
                data_table.add_hline()
                data_table.add_row([" "] + [MultiColumn(c[1], data=c[0].title) for c in merge_repeats(table.header[1])])
                data_table.add_hline()
                data_table.add_row(["Tracker"] + [" " + c.abbreviation + " " for c in table.header[2]])
                data_table.add_hline()
                data_table.end_table_header()
                data_table.add_hline()

                for tracker, data in table.data.items():
                    data_table.add_row([UnsafeCommand("Tracker", [tracker.reference, TRACKER_GROUP])] +
                        [format_cell(x, order[tracker] if not order is None else None) for x, order in zip(data, table.order)])

    if multipart:
        container = Chunk()
        generate_symbols(container, trackers)
        with storage.write("symbols.tex") as out:
            container.dump(out)
        doc.preamble.append(Command("input", "symbols.tex"))
    else:
        generate_symbols(doc.preamble, trackers)

    doc.preamble.append(Command('title', 'VOT toolkit report'))
    doc.preamble.append(Command('author', 'Toolkit version ' + toolkit_version()))
    doc.preamble.append(Command('date', datetime.datetime.now().isoformat()))
    doc.append(NoEscape(r'\maketitle'))

    for key, section in reports.items():

        doc.append(Section(key))

        for item in section:
            if isinstance(item, Table):
                make_table(doc, item)
            if isinstance(item, Plot):
                plot = item
                with doc.create(Figure(position='htbp')) as container:
                    if multipart:
                        plot_name = plot.identifier + ".pdf"
                        with storage.write(plot_name, binary=True) as out:
                            plot.save(out, "PDF")
                        container.add_image(plot_name)
                    else:
                        container.append(insert_figure(plot))
                    container.add_caption(plot.identifier)

                logger.debug("Saving plot %s", item.identifier)
                item.save(key + "_" + item.identifier + '.pdf', "PDF")
            else:
                logger.warning("Unsupported report item type %s", item)

    # TODO: Move to separate function
    #if build:
    #    temp = tempfile.mktemp()
    #    logger.debug("Generating to temporary output %s", temp)
    #    doc.generate_pdf(temp, clean_tex=True)
    #    storage.copy(temp + ".pdf", "report.pdf")
    #else:
    with storage.write("report.tex") as out:
        doc.dump(out)
