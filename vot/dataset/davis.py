"""Dataset adapter for the DAVIS video object segmentation benchmark (train-val).

The loader expects the standard DAVIS directory layout, as extracted from one of the
official archives::

    <root>/
        JPEGImages/<resolution>/<sequence>/00000.jpg ...
        Annotations/<resolution>/<sequence>/00000.png ...
        ImageSets/<year>/{train,val}.txt

Ground truth is stored as single-channel indexed PNG images where each non-zero pixel
value identifies one annotated object. DAVIS 2017 sequences may contain several objects
per frame this way, DAVIS 2016 sequences only ever contain a single object (value 255).
"""

import os
import glob
from collections import defaultdict

import numpy as np

# Preference order when a dataset root supports several variants.
_YEARS = ("2017", "2016")
_RESOLUTIONS = ("480p", "Full-Resolution", "1080p")
_SPLITS = ("val", "train")

_TRAINVAL_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"

def _find_choice(candidates, available):
    """Returns the first candidate that is present in the available set, or None."""
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None

def _read_data(metadata):
    """Reads the sequence data (channels and per-object masks) for a DAVIS sequence.

    :param metadata: Sequence metadata.
    :type metadata: dict

    :returns: Sequence data.
    :rtype: SequenceData"""
    from vot.dataset import DatasetException, PatternFileListChannel, SequenceData, Sequence
    from vot.region import Special
    from vot.region.shapes import Mask
    from vot import config

    from PIL import Image

    images = metadata["images"]
    annotations = metadata["annotations"]

    channels = {}
    # DAVIS frames are numbered starting from 0 (00000.jpg, 00001.jpg, ...)
    channels["color"] = PatternFileListChannel(os.path.join(images, "%05d.jpg"), start=0)
    metadata["channel.default"] = "color"
    metadata["width"], metadata["height"] = channels["color"].size

    length = len(channels["color"])

    annotation_files = sorted(glob.glob(os.path.join(annotations, "*.png")))

    if len(annotation_files) != length:
        raise DatasetException("Number of annotation frames does not match number of "
            "image frames for sequence {}".format(metadata["name"]))

    objects = defaultdict(lambda: [Special(Sequence.UNKNOWN)] * length)

    for index, annotation_file in enumerate(annotation_files):
        label = np.array(Image.open(annotation_file))
        oids = np.unique(label)
        for oid in oids:
            if oid == 0:
                continue
            if oid == 255 and len(oids) > 2: # In case of DAVIS 2016, there is only one object and it is labeled with 255. In case of DAVIS 2017, there may be several objects and the void pixels are labeled with 255.
                # Void pixels
                print("Void pixels in sequence {} frame {}".format(metadata["name"], index))
                objects["_ignore"][index] = Mask((label == 255).astype(np.uint8),
                    optimize=config.mask_optimize_read)
            else:
                objects["object%d" % int(oid)][index] = Mask((label == oid).astype(np.uint8),
                    optimize=config.mask_optimize_read)

    return SequenceData(channels, dict(objects), {}, {}, length)

def read_sequence(path):
    """Reads a DAVIS sequence from the given path. The path is expected to point to the
    per-sequence annotation directory (``Annotations/<resolution>/<sequence>``), the
    corresponding images are located relative to it in the same dataset root.

    :param path: Path to the sequence annotation directory.
    :type path: str

    :returns: Sequence object or None if the path does not contain a valid sequence.
    :rtype: Sequence"""
    from vot.dataset import BasedSequence

    annotations = os.path.normpath(path)
    name = os.path.basename(annotations)
    resolution_dir = os.path.dirname(annotations)
    resolution = os.path.basename(resolution_dir)
    annotations_root = os.path.dirname(resolution_dir)

    if os.path.basename(annotations_root) != "Annotations":
        return None

    root = os.path.dirname(annotations_root)
    images = os.path.join(root, "JPEGImages", resolution, name)

    if not os.path.isdir(annotations) or not os.path.isdir(images):
        return None

    metadata = dict(fps=24, format="davis")
    metadata["name"] = name
    metadata["root"] = root
    metadata["images"] = images
    metadata["annotations"] = annotations
    metadata["channel.default"] = "color"

    return BasedSequence(name, _read_data, metadata)

def list_sequences(path):
    """Lists DAVIS sequences in the given path. The path is expected to be the root of
    an extracted DAVIS archive (containing ``JPEGImages``, ``Annotations`` and
    ``ImageSets`` directories). The newest year and the validation split are preferred
    when several are available.

    :param path: Path to the dataset root.
    :type path: str

    :returns: List of sequence annotation directories or None if not a DAVIS dataset root.
    :rtype: list"""
    if not os.path.isdir(path):
        return None

    imagesets = os.path.join(path, "ImageSets")
    annotations_dir = os.path.join(path, "Annotations")

    if not os.path.isdir(imagesets) or not os.path.isdir(annotations_dir):
        return None

    year = _find_choice(_YEARS, set(os.listdir(imagesets)))
    if year is None:
        return None

    splits = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(imagesets, year))}
    split = _find_choice(_SPLITS, splits)
    if split is None:
        return None

    resolution = _find_choice(_RESOLUTIONS, set(os.listdir(annotations_dir)))
    if resolution is None:
        return None

    split_file = os.path.join(imagesets, year, split + ".txt")
    if not os.path.isfile(split_file):
        return None

    with open(split_file, "r", encoding="utf-8") as filehandle:
        names = [line.strip() for line in filehandle if line.strip()]

    return [os.path.join(annotations_dir, resolution, name) for name in names]

def download_davis_trainval(path):
    """Downloads the DAVIS 2017 train-val (480p) archive to the given path.

    The official archive wraps all dataset content in a top-level ``DAVIS`` directory.
    The archive is extracted to a temporary location and the contents of that wrapper
    directory are moved into ``path`` so that the final layout matches what
    :func:`list_sequences` expects.

    :param path: Path to the dataset folder.
    :type path: str

    :raises DatasetException: If the dataset cannot be downloaded or extracted."""
    import shutil
    import tempfile

    from vot.dataset import DatasetException
    from vot.utilities.net import download_uncompress, NetworkException

    if os.path.isdir(path):
        imagesets = os.path.join(path, "ImageSets")
        annotations_dir = os.path.join(path, "Annotations")

        if os.path.isdir(imagesets) and os.path.isdir(annotations_dir):
            return

    tmp_dir = tempfile.mkdtemp()

    try:
        try:
            download_uncompress(_TRAINVAL_URL, tmp_dir)
        except NetworkException as e:
            raise DatasetException(f"Unable to download DAVIS train-val dataset, please try to download the "
                f"archive manually from {_TRAINVAL_URL} and extract it to {path}") from e
        except IOError as e:
            raise DatasetException("Unable to extract DAVIS train-val dataset, is the target directory "
                "writable and do you have enough space?") from e

        extracted_root = os.path.join(tmp_dir, "DAVIS")
        if not os.path.isdir(extracted_root):
            raise DatasetException("Unexpected DAVIS archive layout, missing top-level DAVIS directory")

        os.makedirs(path, exist_ok=True)
        for name in os.listdir(extracted_root):
            shutil.move(os.path.join(extracted_root, name), os.path.join(path, name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
