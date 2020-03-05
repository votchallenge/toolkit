

from typing import Tuple, List, Union
from abc import ABC, abstractmethod

from matplotlib import colors
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

def show_image(a):
    try:
        import IPython.display
    except ImportError:
        return

    a = np.uint8(a)
    f = BytesIO()
    Image.fromarray(a).save(f, "png")
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

_PALETTE = {
    "white": (1, 1, 1, 1),
    "black": (0, 0, 0, 1),
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
}

def resolve_color(color: Union[Tuple[float, float, float, float], str]):

    if isinstance(color, str):
        return _PALETTE.get(color, (0, 0, 0, 1))
    return (np.clip(color[0], 0, 1), np.clip(color[1], 0, 1), np.clip(color[2], 0, 1), np.clip(color[3], 0, 1))

class DrawHandle(ABC):

    def __init__(self, color: Union[Tuple[float, float, float, float], str] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        self._color = resolve_color(color)
        self._width = width
        self._fill = fill

    def style(self, color: Union[Tuple[float, float, float, float], str] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        self._color = resolve_color(color)
        self._width = width
        self._fill = fill
        return self

    def region(self, region):
        region.draw(self)

    @abstractmethod
    def image(self, image: Union[np.array, Image.Image]):
        pass

    @abstractmethod
    def line(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        pass

    @abstractmethod
    def lines(self, points: List[Tuple[float, float]]):
        pass

    @abstractmethod
    def polygon(self, points: List[Tuple[float, float]]):
        pass

    @abstractmethod
    def mask(self, mask: np.array, offset: Tuple[int, int]):
        pass

class MatplotlibDrawHandle(DrawHandle):

    def __init__(self, axis, color: Tuple[float, float, float, float] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        super().__init__(color, width, fill)
        self._axis = axis

    def image(self, image: Union[np.array, Image.Image]):
        self._axis.imshow(image)

    def line(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self._axis.plot((p1[0], p2[0]), (p1[1], p2[1]), linewidth=self._width, color=self._color)

    def lines(self, points: List[Tuple[float, float]]):
        x = [x for x, _ in points]
        y = [y for _, y in points]
        self._axis.plot(x, y, linewidth=self._width, color=self._color)

    def polygon(self, points: List[Tuple[float, float]]):
        poly = Polygon(points, edgecolor=self._color, linewidth=self._width, fill=self._fill)
        self._axis.add_patch(poly)

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):
        # TODO: segmentation should also have option of non-filled
        cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._color]))
        self._axis.imshow(mask > 0, cmap=cmap, interpolation='none', extent=[offset[0], \
             offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])

class NumpyCanvasDrawHandle(DrawHandle):
    # Does not work at the moment, not implemented
    
    def __init__(self, canvas: np.array, color: Tuple[float, float, float, float] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        super().__init__(color, width, fill)
        self._canvas = canvas

    def image(self, image: Union[np.array, Image.Image]):
        pass

    def line(self, p1, p2):
        cv2.line(self._canvas, p1, p2)

    def lines(self, points: List[Tuple[float, float]]):
        if len(points) == 0:
            return
        p1 = points[0]
        for x, y in points[1:]:
            p2 = (x, y)
            cv2.line(self._canvas, p1, p2)
            p1 = p2

    def polygon(self, points: List[Tuple[float, float]]):
        raise NotImplementedError()

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):
        raise NotImplementedError()
