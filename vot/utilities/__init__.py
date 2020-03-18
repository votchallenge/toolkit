import os, sys
import csv
import hashlib
import errno

from numbers import Number
from typing import Tuple

import six

def import_class(classpath, hints=None):
    delimiter = classpath.rfind(".")
    if delimiter == -1:
        if hints is None:
            hints = []
        for hint in hints:
            try:
                classname = classpath
                module = __import__(hint, globals(), locals(), [classname])
                return getattr(module, classname)
            except ImportError:
                pass
            except TypeError:
                pass
        raise ImportError("Class {} not found in any of paths {}".format(classpath, hints))
    else:
        classname = classpath[delimiter+1:len(classpath)]
        module = __import__(classpath[0:delimiter], globals(), locals(), [classname])
        return getattr(module, classname)

def class_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__

def flip(size: Tuple[Number, Number]) -> Tuple[Number, Number]:
    return (size[1], size[0])

def is_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            raise ImportError("console")
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if 'VSCODE_PID' in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except ImportError:
        return False
    else:
        return True

if is_notebook():
    try:
        from ipywidgets import IntProgress
        from tqdm._tqdm_notebook import tqdm_notebook as tqdm
    except ImportError:
        from tqdm import tqdm
else:
    from tqdm import tqdm

class Progress(tqdm):

  #  def __init__(self, desc=None, total=100):
  #      super().__init__()

    def update_absolute(self, current, total = None):
        if total is not None:
            self.total = total
        self.update(current - self.n)  # will also set self.n = b * bsize
        
    def update_relative(self, n, total = None):
        if total is not None:
            self.total = total
        self.update(n)  # will also set self.n = b * bsize

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
    with open(filename, **open_kwargs) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, escapechar='\\',
                            quoting=csv.QUOTE_NONE)
        return {row[0]: row[1] for row in reader}

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

def arg_hash(*args):
    sha1 = hashlib.sha1()

    for arg in args:
        sha1.update(("(" + str(arg) + ")").encode("utf-8"))

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
        n = bool(val)

        return n
    except ValueError:
        raise RuntimeError("Logical value conversion error")