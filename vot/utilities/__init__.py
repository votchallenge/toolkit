""" This module contains various utility functions and classes used throughout the toolkit. """

import os
import sys
import csv
import re
import hashlib
import logging
import inspect
import time
import concurrent.futures as futures
from logging import Formatter, LogRecord

from numbers import Number
from typing import Any, Mapping, Tuple
import typing
from vot import get_logger

import six
import colorama

__ALIASES = dict()

def import_class(classpath: str) -> typing.Type:
    """Import a class from a string by importing parent packages. 

    Args:
        classpath (str): String representing a canonical class name with all parent packages.

    Raises:
        ImportError: Raised when

    Returns:
        [type]: [description]
    """
    delimiter = classpath.rfind(".")
    if delimiter == -1:
        if classpath in __ALIASES:
            return __ALIASES[classpath]
        else:
            raise ImportError("Class alias '{}' not found".format(classpath))
    else:
        classname = classpath[delimiter+1:len(classpath)]
        module = __import__(classpath[0:delimiter], globals(), locals(), [classname])
        return getattr(module, classname)

def alias(*args):
    """ Decorator for registering class aliases. Aliases are used to refer to classes by a short name.

    Args:
        *args: A list of strings representing aliases for the class.
    """

    def register(cls: typing.Type):
        """ Register the class with the given aliases.  """
        assert cls is not None
        for name in args:
            if name in __ALIASES:
                if not __ALIASES[name] == cls:
                    raise ImportError("Alias already taken")
            else:
                __ALIASES[name] = cls
        return cls
    return register

def class_fullname(o):
    """Returns the full name of the class of the given object.
    
    Args:
        o: The object to get the class name from.
        
    Returns:
        The full name of the class of the given object.
    """
    return class_string(o.__class__)

def class_string(kls):
    """Returns the full name of the given class.

    Args:
        kls: The class to get the name from.

    Returns:
        The full name of the given class.
    """
    assert inspect.isclass(kls)
    module = kls.__module__
    if module is None or module == str.__class__.__module__:
        return kls.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + kls.__name__

def flip(size: Tuple[Number, Number]) -> Tuple[Number, Number]:
    """Flips the given size tuple.

    Args:
        size: The size tuple to flip.
        
    Returns:
        The flipped size tuple.
    """
    return (size[1], size[0])

def flatten(nested_list):
    """Flattens a nested list.

    Args:
        nested_list: The nested list to flatten.

    Returns:
        The flattened list.
    """
    return [item for sublist in nested_list for item in sublist]

from vot.utilities.notebook import is_notebook

if is_notebook():
    try:
        from ipywidgets import IntProgress
        from tqdm._tqdm_notebook import tqdm_notebook as tqdm
    except ImportError:
        from tqdm import tqdm
else:
    from tqdm import tqdm

class Progress(object):
    """Wrapper around tqdm progress bar, enables silecing the progress output and some more
    costumizations.
    """

    class StreamProxy(object):
        """Proxy class for tqdm to enable silent mode."""

        def write(self, x):
            """Write function used by tqdm."""
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x)

        def flush(self):
            """Flush function used by tqdm."""
            #return getattr(self.file, "flush", lambda: None)()
            pass

    @staticmethod
    def logstream():
        """Returns a stream proxy that can be used to redirect output to the progress bar."""
        return Progress.StreamProxy()

    def __init__(self, description="Processing", total=100):
        """Creates a new progress bar.

        Args:
            description: The description of the progress bar.
            total: The total number of steps.
        """
        silent = get_logger().level > logging.INFO

        if not silent:
            self._tqdm = tqdm(disable=False if is_notebook() else None, 
                bar_format=" {desc:20.20} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]", file=sys.stdout, leave=False)
            self._tqdm.desc = description
            self._tqdm.total = total
        if silent or self._tqdm.disable:
            self._tqdm = None
            self._value = 0
            self._total = total if not silent else 0

    def _percent(self, n):
        """Returns the percentage of the given value.

        Args:
            n: The value to compute the percentage of.

        Returns:
            The percentage of the given value.
        """
        return int((n * 100) / self._total)

    def absolute(self, value):
        """Sets the progress to the given value.

        Args:
            value: The value to set the progress to.
        """
        if self._tqdm is None:
            if self._total == 0:
                return
            prev = self._value
            self._value = max(0, min(value, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._tqdm.update(value - self._tqdm.n)  # will also set self.n = b * bsize
        
    def relative(self, n):
        """Increments the progress by the given value.

        Args:
            n: The value to increment the progress by.
        """
        if self._tqdm is None:
            if self._total == 0:
                return
            prev = self._value
            self._value = max(0, min(self._value + n, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._tqdm.update(n)  # will also set self.n = b * bsize 

    def total(self, t):
        """Sets the total number of steps.

        Args:
            t: The total number of steps.
        """
        if self._tqdm is None:
            if self._total == 0:
                return
            self._total = t
        else:
            if self._tqdm.total == t:
                return
            self._tqdm.total = t
            self._tqdm.refresh()

    def __enter__(self):
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context manager."""
        self.close()

    def close(self):
        """Closes the progress bar."""
        if self._tqdm:
            self._tqdm.close()

def extract_files(archive, destination, callback = None):
    """Extracts all files from the given archive to the given destination.

    Args:
        archive: The archive to extract the files from.
        destination: The destination to extract the files to.
        callback: An optional callback function that is called after each file is extracted.
    """
    from zipfile import ZipFile

    with ZipFile(file=archive) as zip_file:
        # Loop over each file
        total=len(zip_file.namelist())
        for file in zip_file.namelist():

            # Extract each file to another directory
            # If you want to extract to current working directory, don't specify path
            zip_file.extract(member=file, path=destination)
            if callback:
                callback(1, total)

def read_properties(filename: str, delimiter: str = '=') -> typing.Dict[str, str]:
    """Reads a given properties file with each line of the format key=value. Returns a dictionary containing the pairs.

    Args:
        filename (str): The name of the file to be read.
        delimiter (str, optional): Key-value delimiter. Defaults to '='.

    Returns:
        [typing.Dict[str, str]]: Resuting properties as a dictionary
    """
    if not os.path.exists(filename):
        return {}
    open_kwargs = {'mode': 'r', 'newline': ''} if six.PY3 else {'mode': 'rb'}
    matcher = re.compile("^([a-zA-Z0-9_\\-.]+) *{} *(.*)$".format(delimiter))
    with open(filename, **open_kwargs) as pfile:
        properties = dict()
        for line in pfile.readlines():
            groups = matcher.match(line.strip())
            if not groups:
                continue
            properties[groups.group(1)] = groups.group(2)
        return properties

def write_properties(filename: str, dictionary: Mapping[str, Any], delimiter: str = '='):
    """Writes the provided dictionary in key sorted order to a properties
        file with each line in the format: key<delimiter>value

    Args:
        filename (str): the name of the file to be written
        dictionary (Mapping[str, str]): a dictionary containing the key/value pairs.
        delimiter (str, optional): _description_. Defaults to '='.
    """

    open_kwargs = {'mode': 'w', 'newline': ''} if six.PY3 else {'mode': 'wb'}
    with open(filename, **open_kwargs) as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, escapechar='\\',
                            quoting=csv.QUOTE_NONE)
        writer.writerows(sorted(dictionary.items()))

def file_hash(filename: str) -> Tuple[str, str]:
    """Calculates MD5 and SHA1 hashes based on file content

    Args:
        filename (str): Filename of the file to open and analyze

    Returns:
        Tuple[str, str]: MD5 and SHA1 hashes as hexadecimal strings.
    """
    
    bufsize = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()
    sha1 = hashlib.sha1()

    with open(filename, 'rb') as f:
        while True:
            data = f.read(bufsize)
            if not data:
                break
            md5.update(data)
            sha1.update(data)

    return md5.hexdigest(), sha1.hexdigest()

def arg_hash(*args, **kwargs) -> str:
    """Computes hash based on input positional and keyword arguments. 

    The algorithm tries to convert all arguments to string, then enclose them with delimiters. The
    positonal arguments are listed as is, keyword arguments are sorted and encoded with their keys as 
    well as values.

    Returns:
        str: SHA1 hash as hexadecimal string
    """
    sha1 = hashlib.sha1()

    for arg in args:
        sha1.update(("(" + str(arg) + ")").encode("utf-8"))

    for (key, val) in sorted(kwargs.items()):
        sha1.update(("(" + str(key) + ":" + str(val) + ")").encode("utf-8"))

    return sha1.hexdigest()

def which(program: str) -> str:
    """Locates an executable in system PATH list by its name.

    Args:
        program (str): Name of the executable

    Returns:
        str: Full path or None if not found
    """

    def is_exe(fpath):
        """Checks if the given path is an executable file.
        
        Args:
            fpath (str): Path to check
            
        Returns:
            bool: True if the path is an executable file
        """
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def normalize_path(path, root=None):
    """Normalizes the given path by making it absolute and removing redundant parts.

    Args:
        path (str): Path to normalize
        root (str, optional): Root path to use if the given path is relative. Defaults to None.

    Returns:
        str: Normalized path
    """
    if os.path.isabs(path):
        return path
    if not root:
        root = os.getcwd()
    return os.path.normpath(os.path.join(root, path))

def localize_path(path):
    """Converts path to local format (backslashes on Windows, slashes on Linux)

    Args:
        path (str): Path to convert

    Returns:
        str: Converted path
    """
    if sys.platform.startswith("win"):
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")

def to_string(n: Any) -> str:
    """Converts object to string, returs empty string if object is None (so a bit different behaviour than
    the original string conversion).

    Args:
        n (Any): Object of any kind

    Returns:
        str: String representation (using built-in conversion)
    """
    if n is None:
        return ""
    else:
        return str(n)

def to_number(val, max_n = None, min_n = None, conversion=int):
    """Converts the given value to a number and checks if it is within the given range. If the value is not a number, 
     a RuntimeError is raised.
      
    Args:
        val (Any): Value to convert
        max_n (int, optional): Maximum allowed value. Defaults to None.
        min_n (int, optional): Minimum allowed value. Defaults to None.
        conversion (function, optional): Conversion function. Defaults to int.
    
    Returns:
        int: Converted value
    """
    try:
        n = conversion(val)

        if not max_n is None:
            if n > max_n:
                raise RuntimeError("Parameter higher than maximum allowed value ({}>{})".format(n, max_n))
        if not min_n is None:
            if n < min_n:
                raise RuntimeError("Parameter lower than minimum allowed value ({}<{})".format(n, min_n))

        return n
    except ValueError:
        raise RuntimeError("Number conversion error")

def to_logical(val):
    """Converts the given value to a logical value (True/False). If the value is not a logical value,
    a RuntimeError is raised.

    Args:
        val (Any): Value to convert

    Returns:
        bool: Converted value
    """
    try:
        if isinstance(val, str):
            return val.lower() in ['true', '1', 't', 'y', 'yes']
        else:
            return bool(val)

    except ValueError:
        raise RuntimeError("Logical value conversion error")

def format_size(num, suffix="B"):
    """Formats the given number as a human-readable size string. 

    Args:
        num (int): Number to format
        suffix (str, optional): Suffix to use. Defaults to "B".

    Returns:
        str: Formatted string
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def singleton(class_):
    """Singleton decorator for classes. 

    Args:
        class_ (class): Class to decorate

    Returns:
        class: Decorated class

    Example:
        @singleton
        class MyClass:
            pass
            
        a = MyClass()
    """
    instances = {}
    def getinstance(*args, **kwargs):
        """Returns the singleton instance of the class. If the instance does not exist, it is created."""
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class ColoredFormatter(Formatter):
    """Colored log formatter using colorama package.
    """

    class Empty(object):
        """An empty class used to copy :class:`~logging.LogRecord` objects without reinitializing them."""

    def __init__(self, **kwargs):
        """Initializes the formatter.

        Args:
            **kwargs: Keyword arguments passed to the base class

        """
        super().__init__(**kwargs)
        colorama.init()

        self._styles = dict(
            debug=colorama.Fore.GREEN,
            verbose=colorama.Fore.BLACK,
            info="",
            notice=colorama.Fore.MAGENTA,
            warning=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            critical=colorama.Fore.RED + colorama.Style.BRIGHT,
        )

    def format(self, record: LogRecord) -> str:
        """Formats message by injecting colorama terminal codes for text coloring.

        Args:
            record (LogRecord): Input log record

        Returns:
            str: Formatted string
        """
        style = self._styles[record.levelname.lower()]

        copy = ColoredFormatter.Empty()
        copy.__class__ = record.__class__
        copy.__dict__.update(record.__dict__)
        msg = record.msg if isinstance(record.msg, str) else str(record.msg)
        copy.msg = style + msg + colorama.Style.RESET_ALL
        record = copy
        # Delegate the remaining formatting to the base formatter.
        return Formatter.format(self, record)


class ThreadPoolExecutor(futures.ThreadPoolExecutor):
    """Thread pool executor with a shutdown method that waits for all threads to finish. 
    """

    def __init__(self, *args, **kwargs):
        """Initializes the thread pool executor."""
        super().__init__(*args, **kwargs)
        #self._work_queue = Queue.Queue(maxsize=maxsize)

    def shutdown(self, wait=True):
        """Shuts down the thread pool executor. If wait is True, waits for all threads to finish.

        Args:
            wait (bool, optional): Wait for all threads to finish. Defaults to True.
        """
        import queue
        with self._shutdown_lock:
            self._shutdown = True
            try:
                while True:
                    item = self._work_queue.get_nowait()
                    item.future.cancel()
            except queue.Empty:
                pass
            self._work_queue.put(None)
        if wait:
            for t in self._threads:
                t.join()

class Timer(object):
    """Simple timer class for measuring elapsed time."""

    def __init__(self, name=None):
        """Initializes the timer.

        Args:
            name (str, optional): Name of the timer. Defaults to None.
        """
        self.name = name

    def __enter__(self):
        """Starts the timer."""
        self._tstart = time.time()

    def __exit__(self, type, value, traceback):
        """Stops the timer and prints the elapsed time."""
        elapsed = time.time() - self._tstart
        if self.name:
            print('[%s]: %.4fs' % (self.name, elapsed))
        else:
            print('Elapsed: %.4fs' % elapsed)