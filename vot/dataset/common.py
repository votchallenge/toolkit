"""This module contains functionality for reading sequences from the storage using VOT compatible format."""

import os
import glob
import logging

import six

import cv2

from vot.dataset import DatasetException, Sequence, BasedSequence, PatternFileListChannel, SequenceData
from vot.region.io import write_trajectory, read_trajectory
from vot.region import Special
from vot.utilities import Progress, localize_path, read_properties, write_properties

logger = logging.getLogger("vot")

def convert_int(value: str) -> int:
    """Converts the given value to an integer. If the value is not a valid integer, None is returned.
    
    Args:
        value (str): The value to convert.
    
    Returns:
        int: The converted value or None if the value is not a valid integer.
    """
    try:
        if value is None:
            return None
        return int(value)
    except ValueError:
        return None
    
def _load_channel(source, length=None):
    """Loads a channel from the given source.

    Args:
        source (str): The source to load the channel from.
        length (int): The length of the channel. If not specified, the channel is loaded from a pattern file list.

    Returns:
        Channel: The loaded channel.
    """

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    return PatternFileListChannel(source, end=length, check_files=length is None)

def _read_data(metadata):
    """Reads data from the given metadata.
    
    Args:
        metadata (dict): The metadata to read data from.
        
    Returns:
        dict: The data read from the metadata.
    """

    channels = {}
    tags = {}
    values = {}
    length = metadata["length"]

    root = metadata["root"]

    for c in ["color", "depth", "ir"]:
        channel_path = metadata.get("channels.%s" % c, None)
        if not channel_path is None:
            channels[c] = _load_channel(os.path.join(root, localize_path(channel_path)), length)

    # Load default channel if no explicit channel data available
    if len(channels) == 0:
        channels["color"] = _load_channel(os.path.join(root, "color", "%08d.jpg"), length=length) 
    else:
        metadata["channel.default"] = next(iter(channels.keys()))

    if metadata.get("width", None) is None or metadata.get("height", None) is None:
        metadata["width"], metadata["height"] = six.next(six.itervalues(channels)).size

    lengths = [len(t) for t in channels.values()]
    assert all([x == lengths[0] for x in lengths]), "Sequence channels have different lengths"
    length = lengths[0]

    objectsfiles = glob.glob(os.path.join(root, 'groundtruth_*.txt'))
    objects = {}
    if len(objectsfiles) > 0:
        for objectfile in objectsfiles:
            groundtruth = read_trajectory(os.path.join(objectfile))
            if len(groundtruth) < length: groundtruth += [Special(Sequence.UNKNOWN)] * (length - len(groundtruth))
            objectid = os.path.basename(objectfile)[12:-4]
            objects[objectid] = groundtruth
    else:
        groundtruth_file = os.path.join(root, metadata.get("groundtruth", "groundtruth.txt"))
        groundtruth = read_trajectory(groundtruth_file)
        if len(groundtruth) < length: groundtruth += [Special(Sequence.UNKNOWN)] * (length - len(groundtruth))
        objects["object"] = groundtruth

    metadata["length"] = length

    tagfiles = glob.glob(os.path.join(root, '*.tag')) + glob.glob(os.path.join(root, '*.label'))

    for tagfile in tagfiles:
        with open(tagfile, 'r') as filehandle:
            tagname = os.path.splitext(os.path.basename(tagfile))[0]
            tag = [line.strip() == "1" for line in filehandle.readlines()]
            while not len(tag) >= length:
                tag.append(False)
            tags[tagname] = tag

    valuefiles = glob.glob(os.path.join(root, '*.value'))

    for valuefile in valuefiles:
        with open(valuefile, 'r') as filehandle:
            valuename = os.path.splitext(os.path.basename(valuefile))[0]
            value = [float(line.strip()) for line in filehandle.readlines()]
            while not len(value) >= length:
                value.append(0.0)
            values[valuename] = value

    for name, tag in tags.items():
        if not len(tag) == length:
            tag_tmp = length * [False]
            tag_tmp[:len(tag)] = tag
            tag = tag_tmp

    for name, value in values.items():
        if not len(value) == length:
            raise DatasetException("Length mismatch for value %s" % name)

    return SequenceData(channels, objects, tags, values, length) 

def _read_metadata(path):
    """Reads metadata from the given path. The metadata is read from the sequence file in the given path.
    
    Args:
        path (str): The path to read metadata from.
        
    Returns:
        dict: The metadata read from the given path.
    """
    metadata = dict(fps=30, format="default")
    metadata["channel.default"] = "color"

    metadata_file = os.path.join(path, 'sequence')
    metadata.update(read_properties(metadata_file))

    metadata["height"] = convert_int(metadata.get("height", None))
    metadata["width"] = convert_int(metadata.get("width", None))
    metadata["length"] = convert_int(metadata.get("length", None))
    metadata["fps"] = convert_int(metadata.get("fps", None))

    metadata["root"] = path

    return metadata

from vot.dataset import sequence_reader, sequence_indexer

@sequence_reader.register("default")
def read_sequence(path):
    """Reads a sequence from the given path.

    Args:
        path (str): The path to read the sequence from.

    Returns:
        Sequence: The sequence read from the given path.
    """
    if not os.path.isfile(os.path.join(path, "sequence")):
        return None

    return BasedSequence(os.path.basename(path), _read_data, _read_metadata(path))

def read_sequence_legacy(path):
    """Reads a sequence from the given path.

    Args:
        path (str): The path to read the sequence from.

    Returns:
        Sequence: The sequence read from the given path.
    """
    if not os.path.isfile(os.path.join(path, "groundtruth.txt")):
        return None

    metadata = dict(fps=30, format="default")
    metadata["channel.default"] = "color"
    metadata["channels.color"] = "%08d.jpg"
    metadata["root"] = path
    metadata["length"] = None

    return BasedSequence(os.path.basename(path), _read_data, metadata=metadata)

@sequence_indexer.register("default")
def index_sequences(path: str) -> None:
    """Indexes the sequences in the given path. Only works if there is a list.txt file in the given path or the path is a list file.
    
    Args:
        path (str): The path to index sequences in.
    """
    names = None

    if os.path.isfile(path):
        with open(os.path.join(path), 'r') as fd:
            names = fd.readlines()

    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "list.txt")):
            with open(os.path.join(path, "list.txt"), 'r') as fd:
                names = fd.readlines()

    if names is None:
        return None

    names = [name.strip() for name in names]
    return names

def download_dataset_meta(url: str, path: str) -> None:
    """Downloads the metadata of a dataset from a given URL and stores it in the given path.
    
    Args:
        url (str): The URL to download the metadata from.
        path (str): The path to store the metadata in.
        
    """
    from vot.utilities.net import download_uncompress, download_json, get_base_url, join_url, NetworkException
    from vot.utilities import format_size

    meta = download_json(url)

    total_size = 0
    for sequence in meta["sequences"]:
        total_size += sequence["annotations"]["uncompressed"]
        for channel in sequence["channels"].values():
            total_size += channel["uncompressed"]

    logger.info('Downloading sequence dataset "%s" with %s sequences (total %s).', meta["name"], len(meta["sequences"]), format_size(total_size))

    base_url = get_base_url(url) + "/"

    failed = []

    with Progress("Downloading", len(meta["sequences"])) as progress:
        for sequence in meta["sequences"]:
            sequence_directory = os.path.join(path, sequence["name"])
            os.makedirs(sequence_directory, exist_ok=True)

            if os.path.isfile(os.path.join(sequence_directory, "sequence")):
                refdata = read_properties(os.path.join(sequence_directory, "sequence"))
                if "uid" in refdata and refdata["uid"] == sequence["annotations"]["uid"]:
                    logger.info('Sequence "%s" already downloaded.', sequence["name"])
                    progress.relative(1)
                    continue

            data = {'name': sequence["name"], 'fps': sequence["fps"], 'format': 'default'}

            annotations_url = join_url(base_url, sequence["annotations"]["url"])

            data["uid"] = sequence["annotations"]["uid"]

            try:
                download_uncompress(annotations_url, sequence_directory)
            except NetworkException as e:
                logger.exception(e)
                failed.append(sequence["name"])
                continue
            except IOError as e:
                logger.exception(e)
                failed.append(sequence["name"])
                continue

            failure = False

            for cname, channel in sequence["channels"].items():
                channel_directory = os.path.join(sequence_directory, cname)
                os.makedirs(channel_directory, exist_ok=True)

                channel_url = join_url(base_url, channel["url"])

                try:
                    download_uncompress(channel_url, channel_directory)
                except NetworkException as e:
                    logger.exception(e)
                    failed.append(sequence["name"])
                    failure = False
                except IOError as e:
                    logger.exception(e)
                    failed.append(sequence["name"])
                    failure = False

                if "pattern" in channel:
                    data["channels." + cname] = cname + os.path.sep + channel["pattern"]
                else:
                    data["channels." + cname] = cname + os.path.sep

                if failure:
                    continue

            write_properties(os.path.join(sequence_directory, 'sequence'), data)
            progress.relative(1)

    if len(failed) > 0:
        logger.error('Failed to download %d sequences.', len(failed))
        logger.error('Failed sequences: %s', ', '.join(failed))
    else:
        logger.info('Successfully downloaded all sequences.')
        with open(os.path.join(path, "list.txt"), "w") as fp:
            for sequence in meta["sequences"]:
                fp.write('{}\n'.format(sequence["name"]))

def write_sequence(directory: str, sequence: Sequence):
    """Writes a sequence to a directory. The sequence is written as a set of images in a directory structure
    corresponding to the channel names. The sequence metadata is written to a file called sequence in the root
    directory.
    
    Args:
        directory (str): The directory to write the sequence to.
        sequence (Sequence): The sequence to write.
    """

    channels = sequence.channels()

    metadata = dict()
    metadata["channel.default"] = sequence.metadata("channel.default", "color")
    metadata["fps"] = sequence.metadata("fps", "30")

    for channel in channels:
        cdir = os.path.join(directory, channel)
        os.makedirs(cdir, exist_ok=True)

        metadata["channels.%s" % channel] = os.path.join(channel, "%08d.jpg")

        for i in range(len(sequence)):
            frame = sequence.frame(i).channel(channel)
            cv2.imwrite(os.path.join(cdir, "%08d.jpg" % (i + 1)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    for tag in sequence.tags():
        data = "\n".join(["1" if tag in sequence.tags(i) else "0" for i in range(len(sequence))])
        with open(os.path.join(directory, "%s.tag" % tag), "w") as fp:
            fp.write(data)

    for value in sequence.values():
        data = "\n".join([ str(sequence.values(i).get(value, "")) for i in range(len(sequence))])
        with open(os.path.join(directory, "%s.value" % value), "w") as fp:
            fp.write(data)

    # Write groundtruth in case of single object
    if len(sequence.objects()) == 1:
        write_trajectory(os.path.join(directory, "groundtruth.txt"), [f.groundtruth() for f in sequence])
    else:
        for id in sequence.objects():
            write_trajectory(os.path.join(directory, "groundtruth_%s.txt" % id), [f.object(id) for f in sequence])

    write_properties(os.path.join(directory, "sequence"), metadata)
