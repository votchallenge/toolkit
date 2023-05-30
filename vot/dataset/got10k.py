""" GOT-10k dataset adapter module. The format of GOT-10k dataset is very similar to a subset of VOT, so there
is a lot of code duplication."""

import os
import glob
import configparser

import six

from vot import get_logger
from vot.dataset import DatasetException, BasedSequence, \
     PatternFileListChannel, SequenceData, Sequence
from vot.region import Special
from vot.region.io import read_trajectory

logger = get_logger()

def load_channel(source):
    """ Load channel from the given source.
    
    Args:
        source (str): Path to the source. If the source is a directory, it is
            assumed to be a pattern file list. If the source is a file, it is
            assumed to be a video file.
            
    Returns:
        Channel: Channel object.
    """
    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    return PatternFileListChannel(source)


def _read_data(metadata):
    """ Read data from the given metadata.
    
    Args:
        metadata (dict): Metadata dictionary.
    """
    channels = {}
    tags = {}
    values = {}
    groundtruth = []

    base = metadata["root"]

    channels["color"] = load_channel(os.path.join(base, "%08d.jpg"))
    metadata["channel.default"] = "color"
    metadata["width"], metadata["height"] = six.next(six.itervalues(channels)).size

    groundtruth_file = os.path.join(base, metadata.get("groundtruth", "groundtruth.txt"))
    groundtruth = read_trajectory(groundtruth_file)

    if len(groundtruth) == 1 and channels["color"].length > 1:
        # We are dealing with testing dataset, only first frame is available, so we pad the
        # groundtruth with unknowns. Only unsupervised experiment will work, but it is ok
        groundtruth.extend([Special(Sequence.UNKNOWN)] * (channels["color"].length - 1))

    metadata["length"] = len(groundtruth)

    tagfiles = glob.glob(os.path.join(base, '*.label'))

    for tagfile in tagfiles:
        with open(tagfile, 'r') as filehandle:
            tagname = os.path.splitext(os.path.basename(tagfile))[0]
            tag = [line.strip() == "1" for line in filehandle.readlines()]
            while not len(tag) >= len(groundtruth):
                tag.append(False)
            tags[tagname] = tag

    valuefiles = glob.glob(os.path.join(base, '*.value'))

    for valuefile in valuefiles:
        with open(valuefile, 'r') as filehandle:
            valuename = os.path.splitext(os.path.basename(valuefile))[0]
            value = [float(line.strip()) for line in filehandle.readlines()]
            while not len(value) >= len(groundtruth):
                value.append(0.0)
            values[valuename] = value

    for name, channel in channels.items():
        if not channel.length == len(groundtruth):
            raise DatasetException("Length mismatch for channel %s" % name)

    for name, tag in tags.items():
        if not len(tag) == len(groundtruth):
            tag_tmp = len(groundtruth) * [False]
            tag_tmp[:len(tag)] = tag
            tag = tag_tmp

    for name, value in values.items():
        if not len(value) == len(groundtruth):
            raise DatasetException("Length mismatch for value %s" % name)

    objects = {"object" : groundtruth}

    return SequenceData(channels, objects, tags, values, len(groundtruth)) 

from vot.dataset import sequence_reader

@sequence_reader.register("GOT-10k")
def read_sequence(path):
    """ Read GOT-10k sequence from the given path.
    
    Args:
        path (str): Path to the sequence.
    """

    if not (os.path.isfile(os.path.join(path, 'groundtruth.txt')) and os.path.isfile(os.path.join(path, 'meta_info.ini'))):
        return None

    metadata = dict(fps=30, format="default")

    if os.path.isfile(os.path.join(path, 'meta_info.ini')):
        config = configparser.ConfigParser()
        config.read(os.path.join(path, 'meta_info.ini'))
        metadata.update(config["METAINFO"])
        metadata["fps"] = int(metadata["anno_fps"][:-3])

    metadata["root"] = path
    metadata["name"] = os.path.basename(path)
    metadata["channel.default"] = "color"

    return BasedSequence(metadata["name"], _read_data, metadata)


