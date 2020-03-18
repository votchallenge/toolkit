import os
from abc import ABC, abstractmethod

from PIL import Image

from vot.dataset import Sequence, VOTSequence, InMemorySequence
from vot.dataset.proxy import FrameMapSequence
from vot.dataset.vot import write_sequence
from vot.region import RegionType
from vot.utilities import to_number, arg_hash

class Transformer(ABC):

    def __init__(self, workspace: "Workspace", **kwargs):
        self._workspace = workspace

    @abstractmethod
    def __call__(self, sequence: Sequence) -> Sequence:
        pass


class Redetection(Transformer):

    def __init__(self, workspace: "Workspace", length: int = 100, initialization: int = 5, padding: float = 2, scaling: float = 1, **kwargs):
        super().__init__(workspace, **kwargs)
        self._workspace = workspace
        self._initialization = to_number(initialization, min_n=1)
        self._length = to_number(length, min_n=self._initialization+1)
        self._padding = to_number(padding, min_n=0, conversion=float)
        self._scaling = to_number(scaling, min_n=0.1, max_n=10, conversion=float)

    def __call__(self, sequence: Sequence) -> Sequence:

        chache_dir = self._workspace.cache(self, arg_hash(sequence.name, self._length, self._initialization, self._padding, self._scaling))

        if not os.path.isfile(os.path.join(chache_dir, "sequence")):
            generated = InMemorySequence(sequence.name, sequence.channels())
            size = (int(sequence.size[0] * self._scaling), int(sequence.size[1] * self._scaling))

            initial_images = dict()
            redetect_images = dict()
            for channel in sequence.channels():
                rect = sequence.frame(0).groundtruth().convert(RegionType.RECTANGLE)

                halfsize = int(max(rect.width, rect.height) * self._scaling / 2)
                x, y = rect.center()

                image = Image.fromarray(sequence.frame(0).image())
                box = (x - halfsize, y - halfsize, x + halfsize, y + halfsize)
                template = image.crop(box)
                initial = Image.new(image.mode, size)
                initial.paste(image, (0, 0))
                redetect = Image.new(image.mode, size)
                redetect.paste(template, (size[0] - template.width, size[1] - template.height))
                initial_images[channel] = initial
                redetect_images[channel] = redetect

            generated.append(initial_images, sequence.frame(0).groundtruth())
            generated.append(redetect_images, sequence.frame(0).groundtruth().move(size[0] - template.width, size[1] - template.height))

            write_sequence(chache_dir, generated)

        source = VOTSequence(chache_dir, name=sequence.name)
        mapping = [0] * self._initialization + [1] * (self._length - self._initialization)
        return FrameMapSequence(source, mapping)
