
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

    def __init__(self, color: Tuple[float, float, float, float] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        self._color = color
        self._width = width
        self._fill = fill

    def style(self, color: Tuple[float, float, float, float] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        self._color = color
        self._width = width
        self._fill = fill
        return self

    def region(self, region):
        region.draw(self)

    @abstractmethod
    def image(self, image):
        pass

    @abstractmethod
    def line(self, x, y):
        pass

    @abstractmethod
    def lines(self, points):
        pass

    @abstractmethod
    def polygon(self, points):
        pass

    @abstractmethod
    def mask(self, mask, offset):
        pass

class MatplotlibDrawHandle(DrawHandle):

    def __init__(self, axis, color: Tuple[float, float, float, float] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        super().__init__(color, width, fill)
        self._axis = axis

    def image(self, image):
        self._axis.imshow(image)

    def line(self, x, y):
        self._axis.plot(x, y, linewidth=self._width, edgecolor=self._color)

    def lines(self, points: List[Tuple[float, float]]):
        pass

    def polygon(self, points: List[Tuple[float, float]]):
        poly = Polygon(points, edgecolor=self._color, linewidth=self._width)
        self._axis.add_patch(poly)

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):

        cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._color]))
        self._axis.imshow(mask > 0, cmap=cmap, interpolation='none', extent=[offset[0], \
             offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])

class NumpyCanvasDrawHandle(DrawHandle):
    # Does not work at the moment, not implemented
    
    def __init__(self, canvas: np.array, color: Tuple[float, float, float, float] = (1, 0, 0, 1), width: int = 1, fill: bool = False):
        super().__init__(color, width, fill)
        self._canvas = canvas

    def image(self, image):
        pass

    def line(self, x, y):
        cv2.line(self._canvas, x, y)
        #self._axis.plot(x, y, linewidth=width, edgecolor=color)

    def lines(self, points: List[Tuple[float, float]]):
        pass

    def polygon(self, points: List[Tuple[float, float]]):
        poly = Polygon(points, edgecolor=self._color, linewidth=self._width)
        self._axis.add_patch(poly)

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):

        cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._color]))
        self._axis.imshow(mask > 0, cmap=cmap, interpolation='none', extent=[offset[0], \
             offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])