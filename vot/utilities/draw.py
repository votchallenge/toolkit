
from typing import Tuple, List
from abc import ABC, abstractmethod

from matplotlib import colors
from matplotlib.patches import Polygon
import PIL.Image
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
    PIL.Image.fromarray(a).save(f, "png")
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

class DrawHandle(ABC):

    @abstractmethod
    def line(self, x, y, width, color):
        pass

    @abstractmethod
    def lines(self, points, width, color):
        pass

    @abstractmethod
    def polygon(self, points, width, color, fill=False):
        pass

    @abstractmethod
    def mask(self, mask, offset, color):
        pass

class MatplotlibDrawHandle(DrawHandle):

    def __init__(self, axis):
        self._axis = axis

    def line(self, x, y, width, color):
        self._axis.plot(x, y, linewidth=width, edgecolor=color)

    def lines(self, points: List[Tuple[float, float]], width: float = 1, \
        color: Tuple[float, float, float, float] = (1, 0, 0, 1)):
        pass

    def polygon(self, points: List[Tuple[float, float]], width: float = 1, \
        color: Tuple[float, float, float, float] = (1, 0, 0, 1), fill=False):
        poly = Polygon(points, edgecolor=color, linewidth=width)
        self._axis.add_patch(poly)

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0), color: Tuple[float, float, float, float] = (1, 0, 0, 1)):

        cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], color]))
        self._axis.imshow(mask > 0, cmap=cmap, interpolation='none', extent=[offset[0], \
             offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])

class NumpyCanvasDrawHandle(DrawHandle):

    def __init__(self, canvas: np.array):
        self._canvas = canvas

    def line(self, x, y, width, color):
        cv2.line(self._canvas, x, y)
        #self._axis.plot(x, y, linewidth=width, edgecolor=color)

    def lines(self, points: List[Tuple[float, float]], width: float = 1, \
        color: Tuple[float, float, float, float] = (1, 0, 0, 1)):
        pass

    def polygon(self, points: List[Tuple[float, float]], width: float = 1, \
        color: Tuple[float, float, float, float] = (1, 0, 0, 1), fill=False):
        poly = Polygon(points, edgecolor=color, linewidth=width)
        self._axis.add_patch(poly)

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0), color: Tuple[float, float, float, float] = (1, 0, 0, 1)):

        cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], color]))
        self._axis.imshow(mask > 0, cmap=cmap, interpolation='none', extent=[offset[0], \
             offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])