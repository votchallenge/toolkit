
import sys, os, time
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

def convert_frame(frame:Frame, channels:list) -> dict:
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
        self._timeout = timeout

        self._watchdog = Thread(target=self._watchdog_loop)
        self._watchdog.start()

        self._watchdog_reset(True)
        self._client = Client(
            streams=(self._process.stdin.fileno(), self._process.stdout.fileno())
        )
        self._watchdog_reset(False)

    def _watchdog_reset(self, enable=True):
        if enable:
            self._watchdog_counter = self._timeout * 10
        else:
            self._watchdog_counter = -1
        
    def _watchdog_loop(self):
        
        while self.alive:
            time.sleep(0.1)
            if self._watchdog_counter < 0:
                continue
            self._watchdog_counter = self._watchdog_counter - 1
            if not self._watchdog_counter:
                self.terminate()

    @property
    def alive(self):
        if not self._process:
            return False
        return self._process.returncode == None

    def initialize(self, frame:Frame, region:Region, properties:dict = dict()) -> Tuple[Region,dict,float]:

        if not self.alive:
            return None

        tlist = convert_frame(frame, self._client.channels)
        tregion = convert_region(region)

        try:
            self._watchdog_reset(True)

            region, properties, elapsed = self._client.initialize(tlist, tregion, properties)

            self._watchdog_reset(False)

            return convert_traxregion(region), properties.dict(), elapsed

        except TraxException as te:
            raise TrackerException(te)

    def frame(self, frame:Frame, properties:dict = dict()) -> Tuple[Region,dict,float]:

        if not self.alive:
            return None

        tlist = convert_frame(frame, self._client.channels)

        try:
            self._watchdog_reset(True)

            region, properties, elapsed = self._client.frame(tlist, properties)

            self._watchdog_reset(False)

            return convert_traxregion(region), properties.dict(), elapsed

        except TraxException as te:
            raise TrackerException(te)

    def terminate(self):
        if not self.alive:
            return

        self._client.quit()

        self._handle = None

        try:
            self._process.wait(3)
        except subprocess.TimeoutExpired:
            pass

        if self._process.returncode == None:
            self._process.terminate()
            try:
                self._process.wait(3)
            except subprocess.TimeoutExpired:
                pass

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

    def initialize(self, frame: Frame, region: Region) -> Tuple[Region,dict,float]:
        self._connect()

        return self._process.initialize(frame, region)

    def update(self, frame: Frame) -> Tuple[Region,dict,float]:
        return self._process.frame(frame)

    def stop(self):
        if self._process:
            self._process.terminate()

def trax_python_adapter(tracker, command, paths, debug: bool=False, linkpaths=[], virtualenv=None):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["sys.path.insert(0, '{}')\n".format(x) for x in paths[::-1]])

    virtualenv_launch = ""
    if virtualenv:
        activate_function = os.path.join(os.path.join(virtualenv, "bin"), "activate_this.py")
        if os.path.isfile(activate_function):
            virtualenv_launch = "exec(open('{}').read(), dict(__file__='{}')) \n".format(activate_function, activate_function)

    command = '{} -c "{}import sys\n{}\nimport {}"'.format(sys.executable, virtualenv_launch, pathimport, command)

    return TraxTrackerRuntime(tracker, command, debug, linkpaths)

def trax_matlab_adapter(tracker, command, paths, debug: bool=False, linkpaths=[]):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["sys.path.import('{}');".format(x) for x in paths])

    command = '{} -c "import sys; {} import {}"'.format(sys.executable, pathimport, command)

    return TraxTrackerRuntime(tracker, command, debug, linkpaths)