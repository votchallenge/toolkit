""" Dataset adapter for the TrackingNet dataset. Note that the dataset is organized a different way than the VOT datasets,
annotated frames are stored in a separate directory. The dataset also contains train and test splits. The loader 
assumes that only one of the splits is used at a time and that the path is given to this part of the dataset. """

import os
import glob
import logging
from collections import OrderedDict

import six

from vot.dataset import Dataset, DatasetException, \
    BasedSequence, PatternFileListChannel, SequenceData, \
    Sequence
from vot.region import Special
from vot.region.io import read_trajectory
from vot.utilities import Progress

logger = logging.getLogger("vot")

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
        source = os.path.join(source, '%d.jpg')
    return PatternFileListChannel(source)


def _read_data(metadata):
    """Internal function for reading data from the given metadata for a TrackingNet sequence.
    
    Args:
        metadata (dict): Metadata dictionary.
    
    Returns:
        SequenceData: Sequence data object.
    """

    channels = {}
    tags = {}
    values = {}
    groundtruth = []

    name = metadata["name"]
    root = metadata["root"]

    channels["color"] = load_channel(os.path.join(root, 'frames', name))
    metadata["channel.default"] = "color"
    metadata["width"], metadata["height"] = six.next(six.itervalues(channels)).size

    groundtruth = read_trajectory(root)

    if len(groundtruth) == 1 and channels["color"].length > 1:
        # We are dealing with testing dataset, only first frame is available, so we pad the
        # groundtruth with unknowns. Only unsupervised experiment will work, but it is ok
        groundtruth.extend([Special(Sequence.UNKNOWN)] * (channels["color"].length - 1))

    metadata["length"] = len(groundtruth)

    objects = {"object" : groundtruth}

    return SequenceData(channels, objects, tags, values, len(groundtruth))

from vot.dataset import sequence_reader

sequence_reader.register("trackingnet")
def read_sequence(path):
    """ Read sequence from the given path. Different to VOT datasets, the sequence is not
    a directory, but a file. From the file name the sequence name is extracted and the
    path to image frames is inferred based on standard TrackingNet directory structure.
    
    Args:
        path (str): Path to the sequence groundtruth.
        
    Returns:
        Sequence: Sequence object.
    """
    if not os.path.isfile(path):
        return None

    name, ext = os.path.splitext(os.path.basename(path))

    if ext != '.txt':
        return None

    root = os.path.dirname(os.path.dirname(os.path.dirname(path)))
 
    if not os.path.isfile(path) and os.path.isdir(os.path.join(root, 'frames', name)):
        return None
    
    metadata = dict(fps=30)
    metadata["channel.default"] = "color"
    metadata["name"] = name
    metadata["root"] = root

    return BasedSequence(name, _read_data, metadata)

from vot.dataset import sequence_indexer

sequence_indexer.register("trackingnet")
def list_sequences(path):
    """ List sequences in the given path. The path is expected to be the root of the TrackingNet dataset split.
    
    Args:
        path (str): Path to the dataset root.
        
    Returns:
        list: List of sequences.
    """
    for dirname in ["anno", "frames"]:
        if not os.path.isdir(os.path.join(path, dirname)):
            return None

    sequences = list(glob.glob(os.path.join(path, "anno", "*.txt")))

    return sequences

   
