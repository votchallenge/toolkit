
import sys
import os
import time
import re
import subprocess
import shutil
import shlex
import socket as socketio
import tempfile
import logging
from typing import Tuple
from threading import Thread, Lock

import colorama

from trax import TraxException
from trax.client import Client
from trax.image import FileImage
from trax.region import Region as TraxRegion
from trax.region import Polygon as TraxPolygon
from trax.region import Mask as TraxMask
from trax.region import Rectangle as TraxRectangle

from vot.dataset import Frame, DatasetException
from vot.region import Region, Polygon, Rectangle, Mask
from vot.tracker import Tracker, TrackerRuntime, TrackerException
from vot.utilities import to_logical, to_number, normalize_path

PORT_POOL_MIN = 9090
PORT_POOL_MAX = 65535

logger = logging.getLogger("vot")

class LogAggregator(object):

    def __init__(self):
        self._fragments = []

    def __call__(self, fragment):
        self._fragments.append(fragment)

    def __str__(self):
        return "".join(self._fragments)

class ColorizedOutput(object):

    def __init__(self):
        colorama.init()

    def __call__(self, fragment):
        print(colorama.Fore.CYAN + fragment + colorama.Fore.RESET, end="")

class PythonCrashHelper(object):

    def __init__(self):
        self._matcher = re.compile(r'''
            ^Traceback
            [\s\S]+?
            (?=^\[|\Z)
            ''', re.M | re.X)

    def __call__(self, log, directory):
        matches = self._matcher.findall(log)
        if len(matches) > 0:
            return matches[-1].group(0)
        return None

def convert_frame(frame: Frame, channels: list) -> dict:
    tlist = dict()

    for channel in channels:
        image = frame.filename(channel)
        if image is None:
            raise DatasetException("Frame does not have information for channel: {}".format(channel))

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
        return Mask(region.array(), region.offset(), optimize=True)

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

def normalize_paths(paths, tracker):
    root = os.path.dirname(tracker.registry)
    return [normalize_path(path, root) for path in paths]

class TrackerProcess(object):

    def __init__(self, command: str, envvars=dict(), timeout=30, log=False, socket=False):
        environment = dict(os.environ)
        environment.update(envvars)

        self._workdir = tempfile.mkdtemp()

        self._returncode = None
        self._socket = None

        if socket:
            for port in range(PORT_POOL_MIN, PORT_POOL_MAX):
                socket = open_local_port(port)
                if not socket is None:
                    self._socket = socket
                    break
            environment["TRAX_SOCKET"] = "{}".format(port)

        logger.debug("Running process: %s", command)

        if sys.platform.startswith("win"):
            self._process = subprocess.Popen(
                    command,
                    cwd=self._workdir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment, bufsize=0)
        else:
            self._process = subprocess.Popen(
                    shlex.split(command),
                    shell=False,
                    cwd=self._workdir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment)

        self._timeout = timeout
        self._client = None

        self._watchdog_lock = Lock()
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
            raise e
        self._watchdog_reset(False)

        self._has_vot_wrapper = not self._client.get("vot") is None

    def _watchdog_reset(self, enable=True):
        if self._watchdog_counter == 0:
            return

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
                logger.warning("Timeout reached, terminating tracker")
                self.terminate()
                break
        print("Terminate")

    @property
    def has_vot_wrapper(self):
        return self._has_vot_wrapper

    @property
    def returncode(self):
        return self._returncode

    @property
    def workdir(self):
        return self._workdir

    @property
    def interrupted(self):
        return self._watchdog_counter == 0

    @property
    def alive(self):
        if self._process is None:
            return False
        self._returncode = self._process.returncode
        return self._returncode is None

    def initialize(self, frame: Frame, region: Region, properties: dict = None) -> Tuple[Region, dict, float]:

        if not self.alive:
            raise TraxException("Tracker not alive")

        if properties is None:
            properties = dict()

        tlist = convert_frame(frame, self._client.channels)
        tregion = convert_region(region)

        self._watchdog_reset(True)

        region, properties, elapsed = self._client.initialize(tlist, tregion, properties)

        self._watchdog_reset(False)

        return convert_traxregion(region), properties.dict(), elapsed


    def frame(self, frame: Frame, properties: dict = dict()) -> Tuple[Region, dict, float]:

        if not self.alive:
            raise TraxException("Tracker not alive")

        tlist = convert_frame(frame, self._client.channels)

        self._watchdog_reset(True)

        region, properties, elapsed = self._client.frame(tlist, properties)

        self._watchdog_reset(False)

        return convert_traxregion(region), properties.dict(), elapsed

    def terminate(self):
        with self._watchdog_lock:

            if not self.alive:
                return

            if not self._client is None:
                self._client.quit()

            try:
                self._process.wait(3)
            except subprocess.TimeoutExpired:
                pass

            if self._process.returncode is None:
                self._process.terminate()
                try:
                    self._process.wait(3)
                except subprocess.TimeoutExpired:
                    pass

                if self._process.returncode is None:
                    self._process.kill()

            if not self._socket is None:
                self._socket.close()

            self._returncode = self._process.returncode

            self._client = None
            self._process = None

    def __del__(self):
        if hasattr(self, "_workdir"):
            shutil.rmtree(self._workdir, ignore_errors=True)

class TraxTrackerRuntime(TrackerRuntime):

    def __init__(self, tracker: Tracker, command: str, log: bool = False, timeout: int = 30, linkpaths=None, envvars=None, arguments=None, socket=False, restart=False, onerror=None):
        super().__init__(tracker)
        self._command = command
        self._process = None
        self._tracker = tracker
        if linkpaths is None:
            linkpaths = []
        if isinstance(linkpaths, str):
            linkpaths = linkpaths.split(os.pathsep)
        linkpaths = normalize_paths(linkpaths, tracker)
        self._socket = to_logical(socket)
        self._restart = to_logical(restart)
        if not log:
            self._output = LogAggregator()
        else:
            self._output = None
        self._timeout = to_number(timeout, min_n=1)
        self._arguments = arguments
        self._onerror = onerror
        self._workdir = None

        if sys.platform.startswith("win"):
            pathvar = "PATH"
        else:
            pathvar = "LD_LIBRARY_PATH"

        envvars[pathvar] = envvars[pathvar] + os.pathsep + os.pathsep.join(linkpaths) if pathvar in envvars else os.pathsep.join(linkpaths)
        envvars["TRAX"] = "1"

        self._envvars = envvars

    @property
    def tracker(self) -> Tracker:
        return self._tracker

    def _connect(self):
        if not self._process:
            if not self._output is None:
                log = self._output
            else:
                log = ColorizedOutput()
            self._process = TrackerProcess(self._command, self._envvars, log=log, socket=self._socket, timeout=self._timeout)
            if self._process.has_vot_wrapper:
                self._restart = True

    def _error(self, exception):
        workdir = None
        timeout = False
        if not self._output is None:
            if not self._process is None:
                if not self._process.alive:
                    self._output("Process exited with code ({})".format(self._process.returncode))
                else:
                    self._output("Process did not finish yet")
                timeout = self._process.interrupted
                self._workdir = self._process.workdir
            else:
                self._output("Process not alive anymore, unable to retrieve return code")

        log = str(self._output)

        try:

            if not self._onerror is None and isinstance(self._onerror, callable):
                self._onerror(log, workdir)

        except Exception as e:
            logger.exception("Error during error handler for runtime of tracker %s", self._tracker.identifier, exc_info=e)

        if timeout:
            raise TrackerException("Tracker interrupted, it did not reply in {} seconds".format(self._timeout), tracker=self._tracker, \
                tracker_log=log if not self._output is None else None)

        raise TrackerException(exception, tracker=self._tracker, \
            tracker_log=log if not self._output is None else None)

    def restart(self):
        try:
            self.stop()
            self._connect()
        except TraxException as e:
            self._error(e)

    def initialize(self, frame: Frame, region: Region, properties: dict = None) -> Tuple[Region, dict, float]:
        try:
            if self._restart:
                self.stop()
            self._connect()

            tproperties = dict(self._arguments)

            if not properties is None:
                tproperties.update(properties)

            return self._process.initialize(frame, region, tproperties)
        except TraxException as e:
            self._error(e)

    def update(self, frame: Frame, properties: dict = None) -> Tuple[Region, dict, float]:
        try:
            if properties is None:
                properties = dict()
            return self._process.frame(frame, properties)
        except TraxException as e:
            self._error(e)

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

def trax_python_adapter(tracker, command, paths, envvars, log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, virtualenv=None, condaenv=None, socket=False, restart=False, **kwargs):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["sys.path.insert(0, '{}');".format(escape_path(x)) for x in normalize_paths(paths[::-1], tracker)])

    interpreter = sys.executable

    if not virtualenv is None and not condaenv is None:
        raise TrackerException("Cannot use both vitrtualenv and conda", tracker=tracker)

    virtualenv_launch = ""
    if not virtualenv is None:
        if sys.platform.startswith("win"):
            activate_function = os.path.join(os.path.join(virtualenv, "Scripts"), "activate_this.py")
            interpreter = os.path.join(os.path.join(virtualenv, "Scripts", "python.exe"))
        else:
            activate_function = os.path.join(os.path.join(virtualenv, "bin"), "activate_this.py")
            interpreter = os.path.join(os.path.join(virtualenv, "bin", "python"))
        if not os.path.isfile(interpreter):
            raise TrackerException("Executable {} not found".format(interpreter), tracker=tracker)

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

    # simple check if the command is only a package name to be imported or a script
    if re.match("^[a-zA-Z_][a-zA-Z0-9_]*$", command) is None:
        # We have to escape all double quotes
        command = command.replace("\"", "\\\"")
    else:
        command = "import " + command

    command = '{} -c "{}import sys;{} {}"'.format(interpreter, virtualenv_launch, pathimport, command)

    envvars["PYTHONUNBUFFERED"] = "1"

    return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)

def trax_matlab_adapter(tracker, command, paths, envvars, log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, socket=False, restart=False, **kwargs):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["addpath('{}');".format(x) for x in normalize_paths(paths, tracker)])

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

    return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)

def trax_octave_adapter(tracker, command, paths, envvars, log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, socket=False, restart=False, **kwargs):
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["addpath('{}');".format(x) for x in normalize_paths(paths, tracker)])

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

    return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)
