import os
import sys
import csv
import re
import hashlib
import errno
import logging
import inspect
import concurrent.futures as futures
from logging import Formatter, LogRecord

from numbers import Number
from typing import Tuple

import six
import colorama

__ALIASES = dict()


def import_class(classpath):
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
    def register(cls):
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
    return class_string(o.__class__)

def class_string(kls):
    assert inspect.isclass(kls)
    module = kls.__module__
    if module is None or module == str.__class__.__module__:
        return kls.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + kls.__name__

def flip(size: Tuple[Number, Number]) -> Tuple[Number, Number]:
    return (size[1], size[0])

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

    class StreamProxy(object):

        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x)
        def flush(self):
            #return getattr(self.file, "flush", lambda: None)()
            pass

    @staticmethod
    def logstream():
        return Progress.StreamProxy()

    def __init__(self, description="Processing", total=100):
        self._tqdm = tqdm(disable=False if is_notebook() else None,
            bar_format=" {desc:20.20} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]")
        self._tqdm.desc = description
        self._tqdm.total = total
        if self._tqdm.disable:
            self._tqdm = None
            self._value = 0
            self._total = total

    def _percent(self, n):
        return int((n * 100) / self._total)

    def absolute(self, value):
        if self._tqdm is None:
            prev = self._value
            self._value = max(0, min(value, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._tqdm.update(value - self._tqdm.n)  # will also set self.n = b * bsize
        
    def relative(self, n):
        if self._tqdm is None:
            prev = self._value
            self._value = max(0, min(self._value + n, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._tqdm.update(n)  # will also set self.n = b * bsize 

    def total(self, t):
        if self._tqdm is None:
            self._total = t
        else:
            if self._tqdm.total == t:
                return
            self._tqdm.total = t
            self._tqdm.refresh()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._tqdm:
            self._tqdm.close()

def extract_files(archive, destination, callback = None):
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

def read_properties(filename, delimiter='='):
    ''' Reads a given properties file with each line of the format key=value.
        Returns a dictionary containing the pairs.
            filename -- the name of the file to be read
    '''
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

def write_properties(filename, dictionary, delimiter='='):
    ''' Writes the provided dictionary in key sorted order to a properties
        file with each line in the format: key<delimiter>value
            filename -- the name of the file to be written
            dictionary -- a dictionary containing the key/value pairs.
    '''
    open_kwargs = {'mode': 'w', 'newline': ''} if six.PY3 else {'mode': 'wb'}
    with open(filename, **open_kwargs) as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, escapechar='\\',
                            quoting=csv.QUOTE_NONE)
        writer.writerows(sorted(dictionary.items()))

def file_hash(filename):

    # BUF_SIZE is totally arbitrary, change for your app!
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

def arg_hash(*args, **kwargs):
    sha1 = hashlib.sha1()

    for arg in args:
        sha1.update(("(" + str(arg) + ")").encode("utf-8"))

    for (key, val) in sorted(kwargs.items()):
        sha1.update(("(" + str(key) + ":" + str(val) + ")").encode("utf-8"))

    return sha1.hexdigest()

def which(program):

    def is_exe(fpath):
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
    if os.path.isabs(path):
        return path
    if not root:
        root = os.getcwd()
    return os.path.normpath(os.path.join(root, path))

def localize_path(path):
    if sys.platform.startswith("win"):
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")

def to_string(n):
    if n is None:
        return ""
    else:
        return str(n)

def to_number(val, max_n = None, min_n = None, conversion=int):
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
    try:
        if isinstance(val, str):
            return val.lower() in ['true', '1', 't', 'y', 'yes']
        else:
            return bool(val)

    except ValueError:
        raise RuntimeError("Logical value conversion error")

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class ColoredFormatter(Formatter):

    class Empty(object):
        """An empty class used to copy :class:`~logging.LogRecord` objects without reinitializing them."""

    def __init__(self, **kwargs):
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


    def format(self, record: LogRecord):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self._work_queue = Queue.Queue(maxsize=maxsize)

    def shutdown(self, wait=True):
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
