
import os
import glob
import tempfile
import six
import logging

import cv2

from vot.dataset import Dataset, DatasetException, Sequence, BaseSequence, PatternFileListChannel, Frame
from vot.region import parse, Region, write_file
from vot.utilities import Progress, extract_files, localize_path, read_properties, write_properties

logger = logging.getLogger("vot")

def load_channel(source):

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    return PatternFileListChannel(source)

class VOTSequence(BaseSequence):

    def __init__(self, base, name=None, dataset=None):
        self._base = base
        if name is None:
            name = os.path.basename(base)
        super().__init__(name, dataset)
        self._metadata["fps"] = 30
        self._metadata["format"] = "default"
        self._metadata["channel.default"] = "color"
        self._scan(base)

    def _scan(self, base):

        metadata_file = os.path.join(base, 'sequence')
        data = read_properties(metadata_file)
        for c in ["color", "depth", "ir"]:
            if "channels.%s" % c in data:
                self._channels[c] = load_channel(os.path.join(self._base, localize_path(data["channels.%s" % c])))

        # Load default channel if no explicit channel data available
        if len(self._channels) == 0:
            self._channels["color"] = load_channel(os.path.join(self._base, "color", "%08d.jpg"))
        else:
            self._metadata["channel.default"] = next(iter(self._channels.keys()))

        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(self._channels)).size

        groundtruth_file = os.path.join(self._base, data.get("groundtruth", "groundtruth.txt"))

        with open(groundtruth_file, 'r') as groundtruth:
            for region in groundtruth.readlines():
                self._groundtruth.append(parse(region))

        self._metadata["length"] = len(self._groundtruth)

        tagfiles = glob.glob(os.path.join(self._base, '*.tag')) + glob.glob(os.path.join(self._base, '*.label'))

        for tagfile in tagfiles:
            with open(tagfile, 'r') as filehandle:
                tagname = os.path.splitext(os.path.basename(tagfile))[0]
                tag = [line.strip() == "1" for line in filehandle.readlines()]
                while not len(tag) >= len(self._groundtruth):
                    tag.append(False)
                self._tags[tagname] = tag
            
        valuefiles = glob.glob(os.path.join(self._base, '*.value'))

        for valuefile in valuefiles:
            with open(valuefile, 'r') as filehandle:
                valuename = os.path.splitext(os.path.basename(valuefile))[0]
                value = [float(line.strip()) for line in filehandle.readlines()]
                while not len(value) >= len(self._groundtruth):
                    value.append(0.0)
                self._values[valuename] = value

        for name, channel in self._channels.items():
            if not channel.length == len(self._groundtruth):
                raise DatasetException("Length mismatch for channel %s" % name)

        for name, tags in self._tags.items():
            if not len(tags) == len(self._groundtruth):
                tag_tmp = len(self._groundtruth) * [False]
                tag_tmp[:len(tags)] = tags
                tags = tag_tmp

        for name, values in self._values.items():
            if not len(values) == len(self._groundtruth):
                raise DatasetException("Length mismatch for value %s" % name)

class VOTDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

        if not os.path.isfile(os.path.join(path, "list.txt")):
            raise DatasetException("Dataset not available locally")

        with open(os.path.join(path, "list.txt"), 'r') as fd:
            names = fd.readlines()
        self._sequences = {name.strip() : VOTSequence(os.path.join(path, name.strip()), dataset=self) for name in Progress(names, desc="Loading dataset", unit="sequences") }

    @property
    def path(self):
        return self._path

    @property
    def length(self):
        return len(self._sequences)

    def __getitem__(self, key):
        return self._sequences[key]

    def __hasitem__(self, key):
        return key in self._sequences

    def __iter__(self):
        return self._sequences.values().__iter__()

    def list(self):
        return self._sequences.keys()

    @classmethod
    def download(self, url, path="."):
        from vot.utilities import write_properties
        from vot.utilities.net import download, download_json, get_base_url, join_url, NetworkException

        def download_uncompress(url, path):
            tmp_file = tempfile.mktemp() + ".zip"
            with Progress(unit='B', desc="Downloading", leave=False) as pbar:
                download(url, tmp_file, pbar.update_absolute)
            with Progress(unit='files', desc="Extracting", leave=True) as pbar:
                extract_files(tmp_file, path, pbar.update_relative)
            os.unlink(tmp_file)


        if os.path.splitext(url)[1] == '.zip':
            logger.info('Downloading sequence bundle from "%s". This may take a while ...', url)

            try:
                download_uncompress(url, path)
            except NetworkException as e:
                raise DatasetException("Unable do download dataset bundle, Please try to download the bundle manually from {} and uncompress it to {}'".format(url, path))
            except IOError as e:
                raise DatasetException("Unable to extract dataset bundle, is the target directory writable and do you have enough space?")

        else:

            meta = download_json(url)

            logger.info('Downloading sequence dataset "%s" with %s sequences.', meta["name"], len(meta["sequences"]))

            base_url = get_base_url(url) + "/"

            for sequence in Progress(meta["sequences"]):
                sequence_directory = os.path.join(path, sequence["name"])
                os.makedirs(sequence_directory, exist_ok=True)

                data = {'name': sequence["name"], 'fps': sequence["fps"], 'format': 'default'}

                annotations_url = join_url(base_url, sequence["annotations"]["url"])

                try:
                    download_uncompress(annotations_url, sequence_directory)
                except NetworkException as e:
                    raise DatasetException("Unable do download annotations bundle")
                except IOError as e:
                    raise DatasetException("Unable to extract annotations bundle, is the target directory writable and do you have enough space?")

                for cname, channel in sequence["channels"].items():
                    channel_directory = os.path.join(sequence_directory, cname)
                    os.makedirs(channel_directory, exist_ok=True)

                    channel_url = join_url(base_url, channel["url"])

                    try:
                        download_uncompress(channel_url, channel_directory)
                    except NetworkException as e:
                        raise DatasetException("Unable do download channel bundle")
                    except IOError as e:
                        raise DatasetException("Unable to extract channel bundle, is the target directory writable and do you have enough space?")

                    if "pattern" in channel:
                        data["channels." + cname] = cname + os.path.sep + channel["pattern"]
                    else:
                        data["channels." + cname] = cname + os.path.sep

                write_properties(os.path.join(sequence_directory, 'sequence'), data)

            with open(os.path.join(path, "list.txt"), "w") as fp:
                for sequence in meta["sequences"]:
                    fp.write('{}\n'.format(sequence["name"]))

def write_sequence(directory: str, sequence: Sequence):
    
    channels = sequence.channels()

    metadata = dict()
    metadata["channel.default"] = sequence.metadata("channel.default", "color")
    metadata["fps"] = sequence.metadata("fps", "30")

    for channel in channels:
        cdir = os.path.join(directory, channel)
        os.makedirs(cdir, exist_ok=True)

        metadata["channels.%s" % channel] = os.path.join(channel, "%08d.jpg")

        for i in range(sequence.length):
            frame = sequence.frame(i).channel(channel)
            cv2.imwrite(os.path.join(cdir, "%08d.jpg" % (i + 1)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    for tag in sequence.tags():
        data = "\n".join(["1" if tag in sequence.tags(i) else "0" for i in range(sequence.length)])
        with open(os.path.join(directory, "%s.tag" % tag), "w") as fp:
            fp.write(data)

    for value in sequence.values():
        data = "\n".join([ str(sequence.values(i).get(value, "")) for i in range(sequence.length)])
        with open(os.path.join(directory, "%s.value" % value), "w") as fp:
            fp.write(data)

    write_file(os.path.join(directory, "groundtruth.txt"), [f.groundtruth() for f in sequence])
    write_properties(os.path.join(directory, "sequence"), metadata)


VOT_DATASETS = {
    "vot2013" : "http://data.votchallenge.net/vot2013/dataset/description.json",
    "vot2014" : "http://data.votchallenge.net/vot2014/dataset/description.json",
    "vot2015" : "http://data.votchallenge.net/vot2015/dataset/description.json",
    "vot-tir2015" : "http://www.cvl.isy.liu.se/research/datasets/ltir/version1.0/ltir_v1_0_8bit.zip",
    "vot2016" : "http://data.votchallenge.net/vot2016/main/description.json",
    "vot-tir2016" : "http://data.votchallenge.net/vot2016/vot-tir2016.zip",
    "vot2017" : "http://data.votchallenge.net/vot2017/main/description.json",
    "vot-st2018" : "http://data.votchallenge.net/vot2018/main/description.json",
    "vot-lt2018" : "http://data.votchallenge.net/vot2018/longterm/description.json",
    "vot-st2019" : "http://data.votchallenge.net/vot2019/main/description.json",
    "vot-lt2019" : "http://data.votchallenge.net/vot2019/longterm/description.json",
    "vot-rgbd2019" : "http://data.votchallenge.net/vot2019/rgbd/description.json",
    "vot-rgbt2019" : "http://data.votchallenge.net/vot2019/rgbtir/meta/description.json",
    "vot-st2020" : "https://data.votchallenge.net/vot2020/shortterm/description.json",
    "vot-rgbt2020" : "http://data.votchallenge.net/vot2020/rgbtir/meta/description.json",
    "test" : "http://data.votchallenge.net/toolkit/test.zip",
    "segmentation" : "http://box.vicos.si/tracking/vot20_test_dataset.zip"
}

def download_dataset(name, path="."):
    if not name in VOT_DATASETS:
        raise ValueError("Unknown dataset")
    VOTDataset.download(VOT_DATASETS[name], path)