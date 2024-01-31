""" OTB dataset adapter module. OTB is one of the earliest tracking benchmarks. It is a collection of 50/100 sequences 
with ground truth annotations. The dataset is available at http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html.
"""

import os
import six

from vot import get_logger
from vot.dataset import BasedSequence, DatasetException, PatternFileListChannel, SequenceData
from vot.utilities import Progress
from vot.region.io import parse_region

logger = get_logger()

_BASE_URL = "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/"

_OTB50_SUBSET = ["Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace", "BlurOwl", "Bolt", "Box",
    "Car1", "Car4", "CarDark", "CarScale", "ClifBar", "Couple", "Crowds", "David", "Deer", "Diving",
    "DragonBaby", "Dudek", "Football", "Freeman4", "Girl", "Human3", "Human4", "Human6", "Human9",
    "Ironman", "Jump", "Jumping", "Liquor", "Matrix", "MotorRolling", "Panda", "RedTeam", "Shaking",
    "Singer2", "Skating1", "Skating2_1", "Skating2_2", "Skiing", "Soccer", "Surfer", "Sylvester", "Tiger2",
    "Trellis", "Walking", "Walking2", "Woman"]

_SEQUENCES = {
    "Basketball": {"attributes": ["IV", "OCC", "DEF", "OPR", "BC"]},
    "Biker": {"attributes": ["SV", "OCC", "MB", "FM", "OPR", "OV", "LR"]},
    "Bird1": {"attributes": ["DEF", "FM", "OV"]},
    "BlurBody": {"attributes": ["SV", "DEF", "MB", "FM", "IPR"]},
    "BlurCar2": {"attributes": ["SV", "MB", "FM"]},
    "BlurFace": {"attributes": ["MB", "FM", "IPR"]},
    "BlurOwl": {"attributes": ["SV", "MB", "FM", "IPR"]},
    "Bolt": {"attributes": ["OCC", "DEF", "IPR", "OPR"]},
    "Box": {"attributes": ["IV", "SV", "OCC", "MB", "IPR", "OPR", "OV", "BC", "LR"]},
    "Car1": {"attributes": ["IV", "SV", "MB", "FM", "BC", "LR"]},
    "Car4": {"attributes": ["IV", "SV"]},
    "CarDark": {"attributes": ["IV", "BC"]},
    "CarScale": {"attributes": ["SV", "OCC", "FM", "IPR", "OPR"]},
    "ClifBar": {"attributes": ["SV", "OCC", "MB", "FM", "IPR", "OV", "BC"]},
    "Couple": {"attributes": ["SV", "DEF", "FM", "OPR", "BC"]},
    "Crowds": {"attributes": ["IV", "DEF", "BC"]},
    "David": {"attributes": ["IV", "SV", "OCC", "DEF", "MB", "IPR", "OPR"], "start": 300, "stop": 770},
    "Deer": {"attributes": ["MB", "FM", "IPR", "BC", "LR"]},
    "Diving": {"attributes": ["SV", "DEF", "IPR"], "stop": 215},
    "DragonBaby": {"attributes": ["SV", "OCC", "MB", "FM", "IPR", "OPR", "OV"]},
    "Dudek": {"attributes": ["SV", "OCC", "DEF", "FM", "IPR", "OPR", "OV", "BC"]},
    "Football": {"attributes": ["OCC", "IPR", "OPR", "BC"]},
    "Freeman4": {"attributes": ["SV", "OCC", "IPR", "OPR"], "start": 1, "stop": 283},
    "Girl": {"attributes": ["SV", "OCC", "IPR", "OPR"]},
    "Human3": {"attributes": ["SV", "OCC", "DEF", "OPR", "BC"]},
    "Human4": {"attributes": ["IV", "SV", "OCC", "DEF"]},
    "Human6": {"attributes": ["SV", "OCC", "DEF", "FM", "OPR", "OV"]},
    "Human9": {"attributes": ["IV", "SV", "DEF", "MB", "FM"]},
    "Ironman": {"attributes": ["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC", "LR"]},
    "Jump": {"attributes": ["SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"]},
    "Jumping": {"attributes": ["MB", "FM"]},
    "Liquor": {"attributes": ["IV", "SV", "OCC", "MB", "FM", "OPR", "OV", "BC"]},
    "Matrix": {"attributes": ["IV", "SV", "OCC", "FM", "IPR", "OPR", "BC"]},
    "MotorRolling": {"attributes": ["IV", "SV", "MB", "FM", "IPR", "BC", "LR"]},
    "Panda": {"attributes": ["SV", "OCC", "DEF", "IPR", "OPR", "OV", "LR"]},
    "RedTeam": {"attributes": ["SV", "OCC", "IPR", "OPR", "LR"]},
    "Shaking": {"attributes": ["IV", "SV", "IPR", "OPR", "BC"]},
    "Singer2": {"attributes": ["IV", "DEF", "IPR", "OPR", "BC"]},
    "Skating1": {"attributes": ["IV", "SV", "OCC", "DEF", "OPR", "BC"]},
    "Skating2_1": {"attributes": ["SV", "OCC", "DEF", "FM", "OPR"], "base": "Skating2", "groundtruth" : "groundtruth_rect.1.txt"},
    "Skating2_2": {"attributes": ["SV", "OCC", "DEF", "FM", "OPR"], "base": "Skating2", "groundtruth" : "groundtruth_rect.2.txt"},
    "Skiing": {"attributes": ["IV", "SV", "DEF", "IPR", "OPR"]},
    "Soccer": {"attributes": ["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"]},
    "Surfer": {"attributes": ["SV", "FM", "IPR", "OPR", "LR"]},
    "Sylvester": {"attributes": ["IV", "IPR", "OPR"]},
    "Tiger2": {"attributes": ["IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "OV"]},
    "Trellis": {"attributes": ["IV", "SV", "IPR", "OPR", "BC"]},
    "Walking": {"attributes": ["SV", "OCC", "DEF"]},
    "Walking2": {"attributes": ["SV", "OCC", "LR"]},
    "Woman": {"attributes": ["IV", "SV", "OCC", "DEF", "MB", "FM", "OPR"]},
    # OTB-100 sequences
    "Bird2": {"attributes": ["OCC", "DEF", "FM", "IPR", "OPR"]},
    "BlurCar1": {"attributes": ["MB", "FM"], "start": 247},
    "BlurCar3": {"attributes": ["MB", "FM"], "start": 3},
    "BlurCar4": {"attributes": ["MB", "FM"], "start": 18},
    "Board": {"attributes": ["SV", "MB", "FM", "OPR", "OV", "BC"], "zeros": 5},
    "Bolt2": {"attributes": ["DEF", "BC"]},
    "Boy": {"attributes": ["SV", "MB", "FM", "IPR", "OPR"]},
    "Car2": {"attributes": ["IV", "SV", "MB", "FM", "BC"]},
    "Car24": {"attributes": ["IV", "SV", "BC"]},	
    "Coke": {"attributes": ["IV", "OCC", "FM", "IPR", "OPR", "BC"]},	
    "Coupon": {"attributes": ["OCC", "BC"]},
    "Crossing": {"attributes": ["SV", "DEF", "FM", "OPR", "BC"]},
    "Dancer": {"attributes": ["SV", "DEF", "IPR", "OPR"]},
    "Dancer2": {"attributes": ["DEF"]},
    "David2": {"attributes": ["IPR", "OPR"]},
    "David3": {"attributes": ["OCC", "DEF", "OPR", "BC"]},
    "Dog": {"attributes": ["SV", "DEF", "OPR"]},
    "Dog1": {"attributes": ["SV", "IPR", "OPR"]},
    "Doll": {"attributes": ["IV", "SV", "OCC", "IPR", "OPR"]},
    "FaceOcc1": {"attributes": ["OCC"]},
    "FaceOcc2": {"attributes": ["IV", "OCC", "IPR", "OPR"]},
    "Fish": {"attributes": ["IV"]},
    "FleetFace": {"attributes": ["SV", "DEF", "MB", "FM", "IPR", "OPR"]},
    "Football1": {"attributes": ["IPR", "OPR", "BC"], "start": 1, "stop": 74},
    "Freeman1": {"attributes": ["SV", "IPR", "OPR"]},
    "Freeman3": {"attributes": ["SV", "IPR", "OPR"], "start": 1, "stop": 460},
    "Girl2": {"attributes": ["SV", "OCC", "DEF", "MB", "OPR"]},
    "Gym": {"attributes": ["SV", "DEF", "IPR", "OPR"]},
    "Human2": {"attributes": ["IV", "SV", "MB", "OPR"]},
    "Human5": {"attributes": ["SV", "OCC", "DEF"]},
    "Human7": {"attributes": ["IV", "SV", "OCC", "DEF", "MB", "FM"]},
    "Human8": {"attributes": ["IV", "SV", "DEF"]},
    "Jogging1": {"attributes": ["OCC", "DEF", "OPR"], "base": "Jogging", "groundtruth" : "groundtruth_rect.1.txt"},
    "Jogging2": {"attributes": ["OCC", "DEF", "OPR"], "base": "Jogging", "groundtruth" : "groundtruth_rect.2.txt"},
    "KiteSurf": {"attributes": ["IV", "OCC", "IPR", "OPR"]},
    "Lemming": {"attributes": ["IV", "SV", "OCC", "FM", "OPR", "OV"]},
    "Man": {"attributes": ["IV"]},
    "Mhyang": {"attributes": ["IV", "DEF", "OPR", "BC"]},
    "MountainBike": {"attributes": ["IPR", "OPR", "BC"]},
    "Rubik": {"attributes": ["SV", "OCC", "IPR", "OPR"]},
    "Singer1": {"attributes": ["IV", "SV", "OCC", "OPR"]},
    "Skater": {"attributes": ["SV", "DEF", "IPR", "OPR"]},
    "Skater2": {"attributes": ["SV", "DEF", "FM", "IPR", "OPR"]},
    "Subway": {"attributes": ["OCC", "DEF", "BC"]},
    "Suv": {"attributes": ["OCC", "IPR", "OV"]},
    "Tiger1": {"attributes": ["IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"]},
    "Toy": {"attributes": ["SV", "FM", "IPR", "OPR"]},
    "Trans": {"attributes": ["IV", "SV", "OCC", "DEF"]},
    "Twinnings": {"attributes": ["SV", "OPR"]},
    "Vase": {"attributes": ["SV", "FM", "IPR"]},
}

def _load_sequence(metadata):
    """Load a sequence from the OTB dataset.
    
    Args:
        metadata (dict): Sequence metadata.
    """

    channels = {}
    groundtruth = []

    attributes = metadata.get("attributes", {})

    channels["color"] = PatternFileListChannel(os.path.join(metadata["path"], "img", "%%0%dd.jpg" % attributes.get("zeros", 4)),
        start=attributes.get("start", 1), end=attributes.get("stop", None))

    metadata["channel.default"] = "color"
    metadata["width"], metadata["height"] = six.next(six.itervalues(channels)).size

    groundtruth_file = os.path.join(metadata["path"], attributes.get("groundtruth", "groundtruth_rect.txt"))

    with open(groundtruth_file, 'r') as filehandle:
        for region in filehandle.readlines():
            groundtruth.append(parse_region(region.replace("\t", ",").replace(" ", ",")))

    metadata["length"] = len(groundtruth)

    if not len(channels["color"]) == len(groundtruth):
        raise DatasetException("Length mismatch between groundtruth and images %d != %d" % (len(channels["color"]), len(groundtruth)))
    
    objects = {"object" : groundtruth}

    return SequenceData(channels, objects, {}, {}, len(groundtruth)) 

from vot.dataset import sequence_reader

@sequence_reader.register("otb")
def read_sequence(path: str):
    """Reads a sequence from OTB dataset. The sequence is identified by the name of the folder and the
    groundtruth_rect.txt file is expected to be present in the folder.
    
    Args:
        path (str): Path to the sequence folder.
    
    Returns:
        Sequence: The sequence object.
    """

    if not os.path.isfile(os.path.join(path, 'groundtruth_rect.txt')):
        return None

    name = os.path.basename(path)

    if name not in _SEQUENCES:
        return None

    metadata =  {"attributes": _SEQUENCES[name], "path": path}
    return BasedSequence(name.strip(), _load_sequence, metadata)

from vot.dataset import dataset_downloader

@dataset_downloader.register("otb50")
def download_otb50(path: str):
    """Downloads OTB50 dataset to the given path.
    
    Args:
        path (str): Path to the dataset folder.
    """
    dataset = _SEQUENCES
    dataset = {k: v for k, v in dataset.items() if k in _OTB50_SUBSET}
    _download_dataset(path, dataset)

@dataset_downloader.register("otb100")
def download_otb100(path: str):
    """Downloads OTB100 dataset to the given path.
    
    Args:
        path (str): Path to the dataset folder.
    """
    dataset = _SEQUENCES
    _download_dataset(path, dataset)

def _download_dataset(path: str, dataset: dict):
    """Downloads the given dataset to the given path.
    
    Args:
        path (str): Path to the dataset folder.
    """

    from vot.utilities.net import download_uncompress, join_url, NetworkException

    with Progress("Downloading", len(dataset)) as progress:
        for name, metadata in dataset.items():
            name = metadata.get("base", name)
            if not os.path.isdir(os.path.join(path, name)):
                try:
                    download_uncompress(join_url(_BASE_URL, "%s.zip" % name), path)
                except NetworkException as ex:
                    raise DatasetException("Unable do download sequence data") from ex
                except IOError as ex:
                    raise DatasetException("Unable to extract sequence data, is the target directory writable and do you have enough space?") from ex

            progress.relative(1)

    # Write sequence list to a list.txt file
    with open(os.path.join(path, "list.txt"), 'w') as filehandle:
        for name in dataset.keys():
            filehandle.write("%s\n" % name)