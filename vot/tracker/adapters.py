import os
import sys
import re

from vot.utilities import normalize_path

def escape_path(path):
    """ Escapes a path. This method is used to escape a path.
    
    Args:
        path: The path to escape.
        
    Returns:
        The escaped path.
    """
    if sys.platform.startswith("win"):
        return path.replace("\\\\", "\\").replace("\\", "\\\\")
    else:
        return path.replace("'", "'\\''")

def normalize_paths(paths, tracker):
    """ Normalizes a list of paths relative to the tracker source."""
    root = os.path.dirname(tracker.source)
    return [normalize_path(path, root) for path in paths]

class PythonAdapter():
    
    def __init__(self, constructor: callable):
        self.constructor = constructor
    
    def __call__(self, tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, python=None, **kwargs):
        """ Creates a Python adapter for a tracker. This method is used to create a Python adapter for a tracker runtime creation.

        Args:
            tracker: The tracker to create the adapter for.
            command: The command to run the tracker.
            envvars: The environment variables to set.
            paths: The paths to add to the Python path.
            log: Whether to log the tracker output.
            timeout: The timeout in seconds.
            linkpaths: The paths to link.
            arguments: The arguments to pass to the tracker.
            python: The Python interpreter to use.
            kwargs: Additional keyword arguments for constructor.

        Returns:
            The tracker runtime object.
        """
        if not isinstance(paths, list):
            paths = paths.split(os.pathsep)

        pathimport = " ".join(["sys.path.insert(0, '{}');".format(escape_path(x)) for x in normalize_paths(paths[::-1], tracker)])
        interpreter = sys.executable if python is None else python

        # simple check if the command is only a package name to be imported or a script
        if re.match("^[a-zA-Z_][a-zA-Z0-9_\\.]*$", command) is None:
            # We have to escape all double quotes
            command = command.replace("\"", "\\\"")
            command = '{} -c "import sys;{} {}"'.format(interpreter, pathimport, command)
        else:
            command = '{} -m {}'.format(interpreter, command)

        envvars["PYTHONPATH"] = os.pathsep.join(normalize_paths(paths[::-1], tracker))   
        envvars["PYTHONUNBUFFERED"] = "1"

        return self.constructor(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, **kwargs)

class MatlabAdapter():
    def __init__(self, constructor: callable):
        self.constructor = constructor

    def __call__(self, tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, matlab=None, **kwargs):
        """ Creates a Matlab adapter for a tracker. This method is used to create a Matlab adapter for a tracker. 

        Args:
            tracker: The tracker to create the adapter for.
            command: The command to run the tracker.
            envvars: The environment variables to set.
            paths: The paths to add to the Matlab path.
            log: Whether to log the tracker output.
            timeout: The timeout in seconds.
            linkpaths: The paths to link.
            arguments: The arguments to pass to the tracker.
            matlab: The Matlab executable to use.
            kwargs: Additional keyword arguments for constructor.

        Returns:
            The tracker runtime object.
        """
        if not isinstance(paths, list):
            paths = paths.split(os.pathsep)

        pathimport = " ".join(["addpath('{}');".format(x) for x in normalize_paths(paths, tracker)])

        if sys.platform.startswith("win"):
            matlabname = "matlab.exe"
            socket = True # We have to use socket connection in this case
        else:
            matlabname = "matlab"
            
        if matlab is None:
            matlabroot = os.getenv("MATLAB_ROOT", None)
            if matlabroot is None:
                testdirs = os.getenv("PATH", "").split(os.pathsep)
                for testdir in testdirs:
                    if os.path.isfile(os.path.join(testdir, matlabname)):
                        matlabroot = os.path.dirname(testdir)
                        break
                if matlabroot is None:
                    raise RuntimeError("Matlab executable not found, set MATLAB_ROOT environmental variable manually.")
            matlab_executable = os.path.join(matlabroot, 'bin', matlabname)
        else:
            matlab_executable = matlab

        if sys.platform.startswith("win"):
            matlab_executable = '"' + matlab_executable + '"'
            matlab_flags = ['-nodesktop', '-nosplash', '-wait', '-minimize']
        else:
            matlab_flags = ['-nodesktop', '-nosplash']

        matlab_script = 'try; diary ''runtime.log''; {}{}; catch ex; disp(getReport(ex)); end; quit;'.format(pathimport, command)

        command = '{} {} -r "{}"'.format(matlab_executable, " ".join(matlab_flags), matlab_script)

        return self.constructor(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, **kwargs)

class OctaveAdapter():
    def __init__(self, constructor: callable):
        self.constructor = constructor

    def __call__(self, tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, **kwargs):
        """ Creates an Octave adapter for a tracker. This method is used to create an Octave adapter for a tracker. 

        Args:
            tracker: The tracker to create the adapter for.
            command: The command to run the tracker.
            envvars: The environment variables to set.
            paths: The paths to add to the Octave path.
            log: Whether to log the tracker output.
            timeout: The timeout in seconds.
            linkpaths: The paths to link.
            arguments: The arguments to pass to the tracker.
            kwargs: Additional keyword arguments.

        Returns:
            The Octave TraX runtime object.
        """

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

        return self.constructor(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, **kwargs)
