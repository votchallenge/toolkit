""" Drawing utilities for visualizing results."""

from typing import Tuple, List, Union

from matplotlib import colors
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import numpy as np
import cv2
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

def show_image(a):
    """Shows an image in the IPython notebook."""
    try:
        import IPython.display
    except ImportError:
        return

    a = np.uint8(a)
    f = BytesIO()
    Image.fromarray(a).save(f, "png")
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

_PALETTE = {
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
    "gray": (0.5, 0.5, 0.5),
}

def resolve_color(color: Union[Tuple[float, float, float], str]):
    """Resolves a color to a tuple of floats. If the color is a string, it is resolved from an internal palette."""

    if isinstance(color, str):
        return _PALETTE.get(color, (0, 0, 0, 1))
    return (np.clip(color[0], 0, 1), np.clip(color[1], 0, 1), np.clip(color[2], 0, 1))

class DrawHandle(object):
    """Base class for drawing handles."""

    def __init__(self, color: Union[Tuple[float, float, float], str] = (1, 0, 0), width: int = 1, fill: bool = False):
        """Initializes the drawing handle.
        
        Args:
            color (tuple or str): Color of the drawing handle.
            width (int): Width of the drawing handle.
            fill (bool): Whether to fill the drawing handle.
        """
        self._color = resolve_color(color)
        self._width = width
        self._fill = fill

    def style(self, color: Union[Tuple[float, float, float], str] = (1, 0, 0), width: int = 1, fill: bool = False) -> 'DrawHandle':
        """Sets the style of the drawing handle. Returns self for chaining.
        
        Args:
            color (tuple or str): Color of the drawing handle.
            width (int): Width of the drawing handle.
            fill (bool): Whether to fill the drawing handle.
            
        Returns:
            self"""
        color = resolve_color(color)
        self._color = (color[0], color[1], color[2], 1)
        self._width = width
        if fill:
            self._fill = (color[0], color[1], color[2], 0.4)
        else:
            self._fill = None
        return self

    def region(self, region):
        """Draws a region."""
        region.draw(self)
        return self

    def image(self, image: Union[np.ndarray, Image.Image], offset: Tuple[int, int] = None):
        """Draws an image at the given offset."""
        return self

    def line(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        """Draws a line between two points."""
        return self

    def lines(self, points: List[Tuple[float, float]]):
        """Draws a line between multiple points."""
        return self

    def polygon(self, points: List[Tuple[float, float]]):
        """Draws a polygon."""
        return self

    def points(self, points: List[Tuple[float, float]]):
        """Draws points."""
        return self

    def rectangle(self, left: float, top: float, right: float, bottom: float):
        """Draws a rectangle. 
        
        The rectangle is defined by the top-left and bottom-right corners.
        """
        self.polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
        return self

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):
        """Draws a mask."""
        return self

class MatplotlibDrawHandle(DrawHandle):
    """Draw handle for Matplotlib. This handle is used for drawing to a Matplotlib axis."""

    def __init__(self, axis, color: Tuple[float, float, float] = (1, 0, 0), width: int = 1, fill: bool = False, size: Tuple[int, int] = None):
        """Initializes a new instance of the MatplotlibDrawHandle class."""
        super().__init__(color, width, fill)
        self._axis = axis
        self._size = size
        if not self._size is None:
            self._axis.set_xlim(left=0, right=self._size[0])
            self._axis.set_ylim(top=0, bottom=self._size[1])


    def image(self, image: Union[np.ndarray, Image.Image], offset: Tuple[int, int] = None):
        """Draws an image at the given offset."""

        if offset is None:
            offset = (0, 0)

        if isinstance(image, np.ndarray):
            width = image.shape[1]
            height = image.shape[0]
        if isinstance(image, Image.Image):
            width = image.size[0]
            height = image.size[1]

        self._axis.imshow(image, extent=[offset[0], \
                offset[0] + width, offset[1] + height, offset[1]])

        return self

    def line(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        """Draws a line between two points."""
        self._axis.plot((p1[0], p2[0]), (p1[1], p2[1]), linewidth=self._width, color=self._color)
        return self

    def lines(self, points: List[Tuple[float, float]]):
        """Draws a line between multiple points."""
        x = [x for x, _ in points]
        y = [y for _, y in points]
        self._axis.plot(x, y, linewidth=self._width, color=self._color)
        return self

    def polygon(self, points: List[Tuple[float, float]]):
        """Draws a polygon."""
        if self._fill:
            poly = Polygon(points, edgecolor=self._color, linewidth=self._width, fill=True, color=self._fill)
        else:
            poly = Polygon(points, edgecolor=self._color, linewidth=self._width, fill=False)
        self._axis.add_patch(poly)
        return self

    def points(self, points: List[Tuple[float, float]]):
        """Draws points."""
        x, y = zip(*points)
        self._axis.plot(x, y, markeredgecolor=self._color, markeredgewidth=self._width, linewidth=0)
        return self

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):
        """Draws a mask."""
        # TODO: segmentation should also have option of non-filled
        mask[mask != 0] = 1
        if self._fill:
            mask = 2 * mask - cv2.erode(mask, kernel=None, iterations=self._width, borderValue=0)
            cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._fill, self._color]))
            self._axis.imshow(mask, cmap=cmap, interpolation='none', extent=[offset[0], \
                offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])
        else:
            mask = mask - cv2.erode(mask, kernel=None, iterations=self._width, borderValue=0)
            cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._color]))
            self._axis.imshow(mask, cmap=cmap, interpolation='none', extent=[offset[0], \
                offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]])

        if not self._size is None:
            self._axis.set_xlim(left=0, right=self._size[0])
            self._axis.set_ylim(top=0, bottom=self._size[1])
        return self


class ImageDrawHandle(DrawHandle):
    """Draw handle for Pillow. This handle is used for drawing to a Pillow image."""

    @staticmethod
    def _convert_color(c, alpha=255):
        """Converts a color from a float tuple to an integer tuple."""
        return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), alpha)

    def __init__(self, image: Union[np.ndarray, Image.Image], color: Tuple[float, float, float] = (1, 0, 0), width: int = 1, fill: bool = False):
        """Initializes a new instance of the ImageDrawHandle class."""
        super().__init__(color, width, fill)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        self._image = image
        self._handle = ImageDraw.Draw(self._image, 'RGBA')

    @property
    def array(self) -> np.ndarray:
        """Returns the image as a numpy array."""
        return np.asarray(self._image)

    @property
    def snapshot(self) -> Image.Image:
        """Returns a snapshot of the current image."""
        return self._image.copy()

    def image(self, image: Union[np.ndarray, Image.Image], offset: Tuple[int, int] = None):
        """Draws an image at the given offset."""
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        if offset is None:
            offset = (0, 0)
        self._image.paste(image, offset)
        return self

    def line(self, p1, p2):
        """Draws a line between two points."""
        color = ImageDrawHandle._convert_color(self._color)
        self._handle.line([p1, p2], fill=color, width=self._width)
        return self

    def lines(self, points: List[Tuple[float, float]]):
        """Draws a line between multiple points."""
        if len(points) == 0:
            return
        color = ImageDrawHandle._convert_color(self._color)
        self._handle.line(points, fill=color, width=self._width)
        return self

    def polygon(self, points: List[Tuple[float, float]]):
        """Draws a polygon."""
        if len(points) == 0:
            return self

        if self._fill:
            color = ImageDrawHandle._convert_color(self._color, alpha=128)
            self._handle.polygon(points, fill=color)

        color = ImageDrawHandle._convert_color(self._color)
        self._handle.line(points + [points[0]], fill=color, width=self._width)
        return self

    def points(self, points: List[Tuple[float, float]]):
        """Draws points."""
        color = ImageDrawHandle._convert_color(self._color)
        for (x, y) in points:
            self._handle.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=self._width)
        return self

    def mask(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):
        """Draws a mask."""
        if mask.size == 0:
            return self

        if self._fill:
            image = Image.fromarray(mask * 128, mode="L")
            color = ImageDrawHandle._convert_color(self._color, 128)
            self._image.paste(color, offset, mask=image)

        image = Image.fromarray((mask - cv2.erode(mask, kernel=None, iterations=self._width, borderValue=0)) * 255, mode="L")
        color = ImageDrawHandle._convert_color(self._color)
        self._handle.bitmap(offset, image, fill=color)

        return self