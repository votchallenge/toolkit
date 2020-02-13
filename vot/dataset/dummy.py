
import os
import json
import glob
import tempfile

from vot.dataset import Dataset, DatasetException, Sequence, Frame, Channel
from vot.region import Rectangle

import cv2
import numpy as np

class SingleFileChannel(Channel):

    def __init__(self, filename, length):
        super().__init__()
        self._length = length
        self._filename = filename

        im = cv2.imread(self._filename)
        self._width = im.shape[1]
        self._height = im.shape[0]
        self._depth = im.shape[2]


    @property
    def length(self):
        return self._length

    def frame(self, index):
        if index < 0 or index >= self.length:
            return None

        bgr = cv2.imread(self._filename)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def size(self):
        return self._width, self._height

    def filename(self, index):
        if index < 0 or index >= self.length:
            return None

        return self._filename

class DummySequence(Sequence):

    def __init__(self, length=100):
        super().__init__("dummy", None)
        self._channels = dict()
        self._base = tempfile.gettempdir()
        self._metadata = {"fps" : 30, "format" : "default",
                          "channel.default": "color", "length" : length}
        self._metadata["name"] = "dummy"
        self._groundtruth = Rectangle(300, 220, 40, 40)
        self.__generate(self._base)

    def __generate(self, base):

        filename = os.path.join(base, "dummy_rgb.jpg")

        image = np.random.normal(15, 5, (480, 640, 3)).astype(np.uint8)
        image[220:260, 300:340, :] = np.random.normal(230, 20, (40, 40, 3)).astype(np.uint8)

        cv2.imwrite(filename, image)

        self._channels["color"] = SingleFileChannel(filename, self.metadata("length"))

        filename = os.path.join(base, "dummy_depth.jpg")

        image = np.ones((480, 640), dtype=np.uint8) * 200
        image[220:260, 300:340] = 10

        cv2.imwrite(filename, image)

        self._channels["depth"] = SingleFileChannel(filename, self.metadata("length"))

        filename = os.path.join(base, "dummy_ir.jpg")

        image = np.zeros((480, 640), dtype=np.uint8)
        image[220:260, 300:340] = 200

        cv2.imwrite(filename, image)

        self._channels["ir"] = SingleFileChannel(filename, self.metadata("length"))

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
            return [self._groundtruth] * self.length
        return self._groundtruth

    def tags(self, index = None):
        return []

    def values(self, index=None):
        return {}

    def size(self):
        return self.channel().size()
    
    @property
    def length(self):
        return self.metadata("length")
