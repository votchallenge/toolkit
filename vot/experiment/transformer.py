import os
from abc import abstractmethod

from PIL import Image

from vot.dataset import Sequence, VOTSequence, InMemorySequence
from vot.dataset.proxy import FrameMapSequence
from vot.dataset.vot import write_sequence
from vot.region import RegionType
from vot.utilities import alias, arg_hash
from vot.utilities.attributes import Attributee, Integer, Float

class Transformer(Attributee):

    def __init__(self, cache: "LocalStorage", **kwargs):
        super().__init__(**kwargs)
        self._cache = cache

    @abstractmethod
    def __call__(self, sequence: Sequence) -> Sequence:
        raise NotImplementedError

@alias("Redetection", "redetection")
class Redetection(Transformer):

    length = Integer(default=100, val_min=1)
    initialization = Integer(default=5, val_min=1)
    padding = Float(default=2, val_min=0)
    scaling = Float(default=1, val_min=0.1, val_max=10)

    def __call__(self, sequence: Sequence) -> Sequence:

        chache_dir = self._cache.directory(self, arg_hash(sequence.name, **self.dump()))

        if not os.path.isfile(os.path.join(chache_dir, "sequence")):
            generated = InMemorySequence(sequence.name, sequence.channels())
            size = (int(sequence.size[0] * self.scaling), int(sequence.size[1] * self.scaling))

            initial_images = dict()
            redetect_images = dict()
            for channel in sequence.channels():
                rect = sequence.frame(0).groundtruth().convert(RegionType.RECTANGLE)

                halfsize = int(max(rect.width, rect.height) * self.scaling / 2)
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
        mapping = [0] * self.initialization + [1] * (self.length - self.initialization)
        return FrameMapSequence(source, mapping)
