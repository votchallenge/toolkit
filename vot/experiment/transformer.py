""" Transformer module for experiments."""

import os
from abc import abstractmethod
import typing

from PIL import Image

from attributee import Attributee, Integer, Float, Boolean, String, List

from vot.dataset import Sequence, InMemorySequence
from vot.dataset.proxy import FrameMapSequence
from vot.dataset.common import write_sequence, read_sequence
from vot.region import RegionType
from vot.utilities import arg_hash
from vot.experiment import transformer_registry

class Transformer(Attributee):
    """Base class for transformers. Transformers are used to generate new modified sequences from existing ones."""

    def __init__(self, cache: "LocalStorage", **kwargs):
        """Initialize the transformer.

        Args:
            cache (LocalStorage): The cache to be used for storing generated sequences.
        """
        super().__init__(**kwargs)
        self._cache = cache

    @abstractmethod
    def __call__(self, sequence: Sequence) -> typing.List[Sequence]:
        """Generate a list of sequences from the given sequence. The generated sequences are stored in the cache if needed.

        Args:
            sequence (Sequence): The sequence to be transformed.
        
        Returns:
            [list]: A list of generated sequences.
        """
        raise NotImplementedError

@transformer_registry.register("singleobject")
class SingleObject(Transformer):
    """Transformer that generates a sequence for each object in the given sequence."""

    trim = Boolean(default=False, description="Trim each generated sequence to a visible subsection for the selected object")

    def __call__(self, sequence: Sequence) -> typing.List[Sequence]:
        """Generate a list of sequences from the given sequence.
        
        Args:
            sequence (Sequence): The sequence to be transformed.
        """
        from vot.dataset.proxy import ObjectFilterSequence
        
        if len(sequence.objects()) == 1:
            return [sequence]
        
        return [ObjectFilterSequence(sequence, id, self.trim) for id in sequence.objects()]
        
@transformer_registry.register("redetection")
class Redetection(Transformer):
    """Transformer that test redetection of the object in the sequence. The object is shown in several frames and then moved to a different location.
    
    This tranformer can only be used with single-object sequences."""

    length = Integer(default=100, val_min=1)
    initialization = Integer(default=5, val_min=1)
    padding = Float(default=2, val_min=0)
    scaling = Float(default=1, val_min=0.1, val_max=10)

    def __call__(self, sequence: Sequence) -> typing.List[Sequence]:
        """Generate a list of sequences from the given sequence.
        
        Args:
            sequence (Sequence): The sequence to be transformed.
        """

        assert len(sequence.objects()) == 1, "Redetection transformer can only be used with single-object sequences."

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

        source = read_sequence(chache_dir)
        mapping = [0] * self.initialization + [1] * (len(source) - self.initialization)
        return [FrameMapSequence(source, mapping)]

@transformer_registry.register("ignore")
class IgnoreObjects(Transformer):
    """Transformer that hides objects with certain ids from the sequence."""

    ids = List(String(), default=[], description="List of ids to be ignored")

    def __call__(self, sequence: Sequence) -> typing.List[Sequence]:
        """Generate a list of sequences from the given sequence.
        
        Args:
            sequence (Sequence): The sequence to be transformed.
        """
        from vot.dataset.proxy import ObjectsHideFilterSequence
        
        return [ObjectsHideFilterSequence(sequence, self.ids)]