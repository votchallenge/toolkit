
import os
import json
import glob
import tempfile
import requests
import six

from vot.dataset import Dataset, DatasetException, Sequence, PatternFileListChannel, Frame

from vot.utilities import Progress, extract_files

def load_channel(source):

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    
    return PatternFileListChannel(source)

class VOTSequence(Sequence):

    def __init__(self, base, name, dataset):
        super().__init__(name, dataset)
        self._base = base
        self._metadata = {"fps" : 30, "format" : "default",
                          "channel.default": "color", "length" : 0}
        self._metadata["name"] = os.path.basename(base)
        self._channels = {}
        self._tags = {}
        self._values = {}
        self._groundtruth = []
        self.__scan(base)

    def __scan(self, base):
        from vot.utilities import read_properties
        from vot.region import parse

        metadata_file = os.path.join(base, 'sequence')
        data = read_properties(metadata_file)
        for c in ["color", "depth", "ir"]:
            if "channel.%s" % c in data:
                self._channels[c] = load_channel(os.path.join(os.path.join(self._base, c),
                                                 data["channel.%s" % c]))

        # Load default channel if no explicit channel data available
        if len(self._channels) == 0:
            self._channels["color"] = load_channel(os.path.join(os.path.join(self._base, "color"),
                                                   "%08d.jpg"))
        else:
            self._metadata["channel.default"] = iter(self._channels.keys()).first()

        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(self._channels)).size()

        groundtruth_file = os.path.join(self._base, data.get("groundtruth", "groundtruth.txt"))

        with open(groundtruth_file, 'r') as groundtruth:
            for region in groundtruth.readlines():
                self._groundtruth.append(parse(region))

        self._metadata["length"] = len(self._groundtruth)

        tagfiles = glob.glob(os.path.join(self._base, '*.tag')) + glob.glob(os.path.join(self._base, '*.label'))

        for tagfile in tagfiles:
            with open(tagfile, 'r') as filehandle:
                tagname = os.path.splitext(os.path.basename(tagfile))[0]
                self._tags[tagname] = [line.strip() == "1" for line in filehandle.readlines()]

        valuefiles = glob.glob(os.path.join(self._base, '*.value'))

        for valuefile in valuefiles:
            with open(valuefile, 'r') as filehandle:
                valuename = os.path.splitext(os.path.basename(valuefile))[0]
                self._values[valuename] = [float(line.strip()) for line in filehandle.readlines()]

        # TODO: validate size and length!


    def metadata(self, name, default=None):
        return self._metadata.get(name, default)

    def channels(self):
        return self._channels

    def channel(self, channel=None):
        if channel is None:
            channel = self.metadata("channel.default")
        return self._channels[channel]

    def frame(self, index):
        return Frame(self, index)
    
    def groundtruth(self, index=None):
        if index is None:
            return self._groundtruth
        return self._groundtruth[index]

    def tags(self, index = None):
        if index is None:
            return self._tags.keys()
        return [t for t in self._tags if t[index]]

    def values(self, index=None):
        if index is None:
            return self._values.keys()
        return {v: sq[index] for v, sq in self._values.items()}

    def size(self):
        return self.channel().size()
    
    @property
    def length(self):
        return len(self._groundtruth)


class VOTDataset(Dataset):
    
    def __init__(self, path):
        super().__init__(path)
        
        if not os.path.isfile(os.path.join(path, "list.txt")):
            raise DatasetException("Dataset not available locally")
 
        with open(os.path.join(path, "list.txt"), 'r') as fd:
            names = fd.readlines()
        self._sequences = { name.strip() : VOTSequence(os.path.join(path, name.strip()), name.strip(), self) for name in Progress(names, desc="Loading dataset", unit="sequences") }
 
    def path(self):
        return self._path

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
            
        
        if os.path.splitext(url) == '.zip':
            print('Downloading sequence bundle from "{}". This may take a while ...'.format(url))
            
            try:
                download_uncompress(url, path)
            except NetworkException as e:
                raise DatasetException("Unable do download dataset bundle, Please try to download the bundle manually from {} and uncompress it to {}'".format(url, path))
            except IOError as e:
                raise DatasetException("Unable to extract dataset bundle, is the target directory writable and do you have enough space?")

        else:
       
            meta = download_json(url)

            print('Downloading sequence dataset "{}" with {} sequences.'.format(meta["name"], len(meta["sequences"])))
        
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
                    
VOT_DATASETS = {
    "vot2013" : "http://data.votchallenge.net/vot2013/dataset/description.json",
    "vot2014" : "http://data.votchallenge.net/vot2014/dataset/description.json"
    
    
}
                             
def download_dataset(name, path="."):
    if not name in VOT_DATASETS:
        raise ValueError("Unknown dataset")
    VOTDataset.download(VOT_DATASETS[name], path)