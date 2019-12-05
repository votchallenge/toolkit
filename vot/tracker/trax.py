
import sys, os
import subprocess, shlex
from typing import Tuple
from threading import Thread, RLock

from trax import TraxException, TraxStatus
from trax.client import Client
from trax.image import FileImage, MemoryImage, BufferImage
from trax.region import Region as TraxRegion
from trax.region import Polygon as TraxPolygon
from trax.region import Mask as TraxMask
from trax.region import Rectangle as TraxRectangle

from vot.dataset import Frame
from vot.region import Region, Polygon, Rectangle, Special, Mask
from vot.tracker import Tracker, TrackerRuntime, TrackerTimeoutException, TrackerException

def convert_frame(frame:Frame) -> dict:
    channels = ["color", "depth", "ir"]
    tlist = dict()

    for channel in channels:
        image = frame.filename(channel)
        if image is None:
            continue

        tlist[channel] = FileImage.create(image)
    return tlist

def convert_region(region:Region) -> TraxRegion:
    if isinstance(region, Rectangle):
        return TraxRectangle.create(region.x, region.y, region.width, region.height)
    elif isinstance(region, Polygon):
        return TraxPolygon.create(region.points)
    elif isinstance(region, Mask):
        return TraxMask.create(region.bitmask, x=region.offset[0], y=region.offset[1])

    return None

def convert_traxregion(region:TraxRegion) -> Region:
    if region.type == TraxRegion.RECTANGLE:
        return Rectangle(region.x, region.y, region.width, region.height)
    elif isinstance(region, Polygon):
        return Polygon(region.points)

    return None

class TrackerProcess(object):

    def __init__(self, command: str, envvars = dict(), timeout=30):
        self._arguments = shlex.split(command)
        self._process = subprocess.Popen(
                        self._arguments, 
                        shell=False, 
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, 
                        env=envvars)

        self._client = Client(
            streams=(self._process.stdin.fileno(), self._process.stdout.fileno())
        )

 #       self._watchdog = Thread(self._watchdog_loop)
 #       self._watchdog.start()

 #   def _watchdog_loop(self):
 #       while True:
 #           pass

    @property
    def alive(self):
        if not self._process:
            return False
        return self._process.returncode == None

    def initialize(self, frame:Frame, region:Region, properties:dict = dict()) -> Tuple[Region,dict]:

        if not self.alive:
            return None

        tlist = convert_frame(frame)
        tregion = convert_region(region)

        try:
            region, properties = self._client.initialize(tlist, tregion, properties)

            return convert_traxregion(region), properties.dict()

        except TraxException as te:
            raise TrackerException(te)

    def frame(self, frame:Frame, properties:dict = dict()) -> Region:

        if not self.alive:
            return None

        tlist = convert_frame(frame)

        try:
            region, properties = self._client.frame(tlist, properties)

            return convert_traxregion(region), properties.dict()

        except TraxException as te:
            raise TrackerException(te)

    def terminate(self):
        if not self.alive:
            return
        self._handle = None
        self._process.wait(3)

        if self._process.returncode == None:
            self._process.terminate()
            self._process.wait(3)

            if self._process.returncode == None:
                self._process.kill()

        self._process = None

class TraxTrackerRuntime(TrackerRuntime):

    def __init__(self, tracker:Tracker, command: str, debug: bool=False, linkpaths=[]):
        self._command = command
        self._debug = debug
        self._process = None
        self._tracker = tracker
        if isinstance(linkpaths, str):
            self._linkpaths = linkpaths.split(os.pathsep)
        else:
            self._linkpaths = linkpaths

    def tracker(self) -> Tracker:
        return self._tracker

    def _connect(self):
        if not self._process:
            envvars = dict()
            if sys.platform == "win32":
                envvars["PATH"] = os.pathsep.join(self._linkpaths)
            else:
                envvars["LD_LIBRARY_PATH"] = os.pathsep.join(self._linkpaths)
            envvars["TRAX"] = "1"
            self._process = TrackerProcess(self._command, envvars)

    def restart(self):
        if self._process:
            self._process.terminate()
        self._connect()

    def initialize(self, frame: Frame, region: Region) -> Tuple[Region, dict]:
        self._connect()
        return self._process.initialize(frame, region)

    def update(self, frame: Frame) -> Tuple[Region, dict]:
        return self._process.frame(frame)


def trax_python_adapter(tracker, command, paths, debug: bool=False, linkpaths=[], virtualenv=None):
    pathimport = " ".join(["sys.path.import('{}');".format(x) for x in paths])

    virtualenv_launch = ""
    if virtualenv:
        activate_function = os.path.join(os.path.join(virtualenv, "bin"), "activate_this.py")
        if os.path.isfile(activate_function):
            virtualenv_launch = "execfile('{}', dict(__file__='{}');".format(activate_function, activate_function)

    command = '{} -c "{} import sys; {} import {}"'.format(virtualenv_launch, sys.executable, pathimport, command)

    return TraxTrackerRuntime(command, debug, linkpaths)

def trax_matlab_adapter(tracker, command, paths, debug: bool=False, linkpaths=[]):
    pathimport = " ".join(["sys.path.import('{}');".format(x) for x in paths])

    command = '{} -c "import sys; {} import {}"'.format(sys.executable, pathimport, command)

    return TraxTrackerRuntime(command, debug, linkpaths)