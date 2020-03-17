
import os
import math
import tempfile

from vot.dataset import VOTSequence
from vot.region import Rectangle, write_file
from vot.utilities import write_properties

from PIL import Image
import numpy as np

class DummySequence(VOTSequence):

    def __init__(self, length=100, size=(640, 480)):
        base = os.path.join(tempfile.gettempdir(), "vot_dummy_%d_%d_%d" % (length, size[0], size[1]))
        if not os.path.isdir(base) or not os.path.isfile(os.path.join(base, "groundtruth.txt")):
            DummySequence._generate(base, length, size)
        super().__init__(base, None)

    @staticmethod
    def _generate(base, length, size):

        background_color = Image.fromarray(np.random.normal(15, 5, (size[1], size[0], 3)).astype(np.uint8))
        background_depth = Image.fromarray(np.ones((size[1], size[0]), dtype=np.uint8) * 200)
        background_ir = Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8))

        template = Image.open(os.path.join(os.path.dirname(__file__), "cow.png"))

        dir_color = os.path.join(base, "color")
        dir_depth = os.path.join(base, "depth")
        dir_ir = os.path.join(base, "ir")

        os.makedirs(dir_color, exist_ok=True)
        os.makedirs(dir_depth, exist_ok=True)
        os.makedirs(dir_ir, exist_ok=True)

        path_color = os.path.join(dir_color, "%08d.jpg")
        path_depth = os.path.join(dir_depth, "%08d.png")
        path_ir = os.path.join(dir_ir, "%08d.png")

        groundtruth = []

        center_x = size[0] / 2
        center_y = size[1] / 2

        radius = min(center_x - template.size[0], center_y - template.size[1])

        speed = (math.pi * 2) / length

        for i in range(length):
            frame_color = background_color.copy()
            frame_depth = background_depth.copy()
            frame_ir = background_ir.copy()

            x = int(center_x + math.cos(i * speed) * radius - template.size[0] / 2)
            y = int(center_y + math.sin(i * speed) * radius - template.size[1] / 2)

            frame_color.paste(template, (x, y), template)
            frame_depth.paste(10, (x, y), template)
            frame_ir.paste(240, (x, y), template)

            frame_color.save(path_color % (i + 1))
            frame_depth.save(path_depth % (i + 1))
            frame_ir.save(path_ir % (i + 1))

            groundtruth.append(Rectangle(x, y, template.size[0], template.size[1]))

        write_file(os.path.join(base, "groundtruth.txt"), groundtruth)
        metadata = {"name": "dummy", "fps" : 30, "format" : "dummy",
                          "channel.default": "color"}
        write_properties(os.path.join(base, "sequence"), metadata)

