
import sys, os, time
import subprocess, shlex
import socket as socketio
from typing import Tuple
from threading import Thread

from trax import TraxException
from trax.client import Client
from trax.image import FileImage, MemoryImage, BufferImage
from trax.region import Region as TraxRegion
from trax.region import Polygon as TraxPolygon
from trax.region import Mask as TraxMask
from trax.region import Rectangle as TraxRectangle

from vot.dataset import Frame
from vot.region import Region, Polygon, Rectangle, Mask
from vot.tracker import Tracker, TrackerRuntime, TrackerException


PORT_POOL_MIN = 9090
PORT_POOL_MAX = 65535

def convert_frame(frame: Frame, channels: list) -> dict:
    tlist = dict()

    for channel in channels:
        image = frame.filename(channel)
        if image is None:
            continue

        tlist[channel] = FileImage.create(image)
    return tlist

def convert_region(region: Region) -> TraxRegion:
    if isinstance(region, Rectangle):
        return TraxRectangle.create(region.x, region.y, region.width, region.height)
    elif isinstance(region, Polygon):
        return TraxPolygon.create(region.points)
    elif isinstance(region, Mask):
        return TraxMask.create(region.mask, x=region.offset[0], y=region.offset[1])

    return None

def convert_traxregion(region: TraxRegion) -> Region:
    if region.type == TraxRegion.RECTANGLE:
        x, y, width, height = region.bounds()
        return Rectangle(x, y, width, height)
    elif region.type == TraxRegion.POLYGON:
        return Polygon(list(region))
    elif region.type == TraxRegion.MASK:
        return Mask(region.array(), region.offset())

    return None

def open_local_port(port: int):
    socket = socketio.socket(socketio.AF_INET, socketio.SOCK_STREAM)
    try:
        socket.setsockopt(socketio.SOL_SOCKET, socketio.SO_REUSEADDR, 1)
        socket.bind(('127.0.0.1', port))
        socket.listen(1)
        return socket
    except OSError:
        try:
            socket.close()
        except OSError:
            pass
        return None

class TrackerProcess(object):

    def __init__(self, command: str, envvars=dict(), timeout=30, log=False, socket=False):
        environment = dict(os.environ)
        environment.update(envvars)

        self._socket = None

        if socket:
            for port in range(PORT_POOL_MIN, PORT_POOL_MAX):
                socket = open_local_port(port)
                if not socket is None:
                    self._socket = socket
                    break
            environment["TRAX_SOCKET"] = "{}".format(port)

        if sys.platform.startswith("win"):
            self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment, bufsize=0)
        else:
            self._process = subprocess.Popen(
                    shlex.split(command),
                    shell=False,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment)

        self._timeout = timeout
        self._client = None

        self._watchdog_counter = 0
        self._watchdog = Thread(target=self._watchdog_loop)
        self._watchdog.start()

        self._watchdog_reset(True)

        try:
            if socket:
                self._client = Client(stream=self._socket.fileno(), timeout=30, log=log)
            else:
                self._client = Client(
                    stream=(self._process.stdin.fileno(), self._process.stdout.fileno()), log=log
                )
        except TraxException as e:
            self.terminate()
            self._watchdog_reset(False)
            raise TrackerException(e)
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
                break

    @property
    def alive(self):
        if not self._process:
            return False
        return self._process.returncode == None

    def initialize(self, frame: Frame, region: Region, properties: dict = dict()) -> Tuple[Region, dict, float]:

        if not self.alive:
            raise TrackerException("Tracker not alive")

        tlist = convert_frame(frame, self._client.channels)
        tregion = convert_region(region)

        try:
            self._watchdog_reset(True)

            region, properties, elapsed = self._client.initialize(tlist, tregion, properties)

            self._watchdog_reset(False)

            return convert_traxregion(region), properties.dict(), elapsed

        except TraxException as te:
            raise TrackerException(te)

    def frame(self, frame: Frame, properties: dict = dict()) -> Tuple[Region, dict, float]:

        if not self.alive:
            raise TrackerException("Tracker not alive")

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

        if not self._client is None:
            self._client.quit()
            self._client = None

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

        if not self._socket is None:
            self._socket.close()

        self._process = None

class TraxTrackerRuntime(TrackerRuntime):

    def __init__(self, tracker: Tracker, command: str, log: bool = False, linkpaths=[], envvars=None, socket=False, restart=True):
        self._command = command
        self._log = log
        self._process = None
        self._tracker = tracker
        if isinstance(linkpaths, str):
            linkpaths = linkpaths.split(os.pathsep)
        self._socket = socket
        self._restart = restart

        if sys.platform.startswith("win"):
            pathvar = "PATH"
        else:
            pathvar = "LD_LIBRARY_PATH"

        envvars[pathvar] = envvars[pathvar] + os.pathsep + os.pathsep.join(linkpaths) if pathvar in envvars else os.pathsep.join(linkpaths)
        envvars["TRAX"] = "1"

        self._envvars = envvars

    def tracker(self) -> Tracker:
        return self._tracker

    def _connect(self):
        if not self._process:
            self._process = TrackerProcess(self._command, self._envvars, log=self._log, socket=self._socket)

    def restart(self):
        self.stop()
        self._connect()

    def initialize(self, frame: Frame, region: Region) -> Tuple[Region, dict, float]:
        if self._restart:
            self.restart()
        else:
            self._connect()
        return self._process.initialize(frame, region)

    def update(self, frame: Frame) -> Tuple[Region, dict, float]:
        return self._process.frame(frame)

    def stop(self):
        if not self._process is None:
            self._process.terminate()
            self._process = None

    def __del__(self):
        if not self._process is None:
            self._process.terminate()
            self._process = None

def escape_path(path):
    if sys.platform.startswith("win"):
        return path.replace("\\\\", "\\").replace("\\", "\\\\")
    else:
        return path

def trax_python_adapter(tracker, command, paths, envvars, log: bool = False, linkpaths=[], virtualenv=None, condaenv=None, socket=False, **kwargs):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["sys.path.insert(0, '{}');".format(escape_path(x)) for x in paths[::-1]])

    interpreter = sys.executable

    if not virtualenv is None and not condaenv is None:
        raise TrackerException("Cannot use both vitrtualenv and Conda")

    virtualenv_launch = ""
    if not virtualenv is None:
        if sys.platform.startswith("win"):
            activate_function = os.path.join(os.path.join(virtualenv, "Scripts"), "activate_this.py")
            interpreter = os.path.join(os.path.join(virtualenv, "Scripts", "python.exe"))
        else:
            activate_function = os.path.join(os.path.join(virtualenv, "bin"), "activate_this.py")
            interpreter = os.path.join(os.path.join(virtualenv, "bin", "python"))
        if not os.path.isfile(interpreter):
            raise TrackerException("Executable {} not found".format(interpreter))

        if os.path.isfile(activate_function):
            virtualenv_launch = "exec(open('{0}').read(), dict(__file__='{0}'));".format(escape_path(activate_function))

    if not condaenv is None:
        if sys.platform.startswith("win"):
            paths = ["Library\\mingw-w64\\bin", "Library\\usr\\bin", "Library\\bin", "Scripts", "bin"]
            interpreter = os.path.join(os.path.join(virtualenv, "python.exe"))
        else:
            paths = [] #TODO
            interpreter = os.path.join(os.path.join(virtualenv, "python"))
        paths = [os.path.join(condaenv, x) for x in paths]
        envvars["PATH"] = os.pathsep.join(paths) + os.pathsep + envvars.get("PATH", "")

        if os.path.isfile(activate_function):
            virtualenv_launch = "exec(open('{0}').read(), dict(__file__='{0}'));".format(escape_path(activate_function))

    command = '{} -c "{}import sys;{} import {}"'.format(interpreter, virtualenv_launch, pathimport, command)

    return TraxTrackerRuntime(tracker, command, log, linkpaths, envvars, socket)

def trax_matlab_adapter(tracker, command, paths, envvars, log: bool = False, linkpaths=[], socket=False, **kwargs):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["addpath('{}');".format(x) for x in paths])

    matlabroot = os.getenv("MATLAB_ROOT", None)

    if sys.platform.startswith("win"):
        matlabname = "matlab.exe"
        socket = True # We have to use socket connection in this case
    else:
        matlabname = "matlab"

    if matlabroot is None:
        testdirs = os.getenv("PATH", "").split(os.pathsep)
        for testdir in testdirs:
            if os.path.isfile(os.path.join(testdir, matlabname)):
                matlabroot = os.path.dirname(testdir)
                break
        if matlabroot is None:
            raise RuntimeError("Matlab executable not found, set MATLAB_ROOT environmental variable manually.")

    if sys.platform.startswith("win"):
        matlab_executable = '"' + os.path.join(matlabroot, 'bin', matlabname) + '"'
        matlab_flags = ['-nodesktop', '-nosplash', '-wait', '-minimize']
    else:
        matlab_executable = os.path.join(matlabroot, 'bin', matlabname)
        matlab_flags = ['-nodesktop', '-nosplash']

    matlab_script = 'try; diary ''runtime.log''; {}{}; catch ex; disp(getReport(ex)); end; quit;'.format(pathimport, command)

    command = '{} {} -r "{}"'.format(matlab_executable, " ".join(matlab_flags), matlab_script)

    return TraxTrackerRuntime(tracker, command, log, linkpaths, envvars, socket)

def trax_octave_adapter(tracker, command, paths, envvars, log: bool = False, linkpaths=[], socket=False, **kwargs):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["addpath('{}');".format(x) for x in paths])

    octaveroot = os.getenv("OCTAVE_ROOT", None)

    if sys.platform.startswith("win"):
        octavename = "octave.exe"
    else:
        octavename = "octave"

    if octaveroot is None:
        testdirs = os.getenv("PATH", "").split(os.pathsep)
        for testdir in testdirs:
            if os.path.isfile(os.path.join(testdir, octavename)):
                octaveroot = os.path.dirname(testdir)
                break
        if octaveroot is None:
            raise RuntimeError("Octave executable not found, set OCTAVE_ROOT environmental variable manually.")

    if sys.platform.startswith("win"):
        octave_executable = '"' + os.path.join(octaveroot, 'bin', octavename) + '"'
    else:
        octave_executable = os.path.join(octaveroot, 'bin', octavename)

    octave_flags = ['--no-gui', '--no-window-system']

    octave_script = 'try; diary ''runtime.log''; {}{}; catch ex; disp(ex.message); for i = 1:size(ex.stack) disp(''filename''); disp(ex.stack(i).file); disp(''line''); disp(ex.stack(i).line); endfor; end; quit;'.format(pathimport, command)

    command = '{} {} --eval "{}"'.format(octave_executable, " ".join(octave_flags), octave_script)

    return TraxTrackerRuntime(tracker, command, log, linkpaths, envvars, socket)

