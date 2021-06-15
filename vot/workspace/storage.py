
import os
import pickle

from abc import ABC, abstractmethod
import typing

import cachetools

from attributee.object import class_fullname

from ..experiment import Experiment
from ..dataset import Sequence
from ..tracker import Tracker, Results

class Storage(ABC):
    """Abstract superclass for workspace storage abstraction
    """

    @abstractmethod
    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence) -> Results:
        """Returns results object for the given tracker, experiment, sequence combination

        Args:
            tracker (Tracker): Selected tracker
            experiment (Experiment): Selected experiment
            sequence (Sequence): Selected sequence
        """
        pass

    @abstractmethod
    def documents(self) -> typing.List[str]:
        """Lists documents in the storage.
        """
        pass

    @abstractmethod
    def folders(self) -> typing.List[str]:
        """Lists folders in the storage.
        """
        pass

    @abstractmethod
    def write(self, name:str, binary: bool = False):
        """Opens the given file entry for writing, returns opened handle.

        Args:
            name (str): File name.
            binary (bool, optional): Open file in binary mode. Defaults to False.
        """
        pass

    @abstractmethod
    def read(self, name, binary=False):
        """Opens the given file entry for reading, returns opened handle.

        Args:
            name (str): File name.
            binary (bool, optional): Open file in binary mode. Defaults to False.
        """
        pass

    @abstractmethod
    def isdocument(self, name: str) -> bool:
        """Checks if given name is a document/file in this storage.

        Args:
            name (str): Name of the entry to check

        Returns:
            bool: Returns True if entry is a document, False otherwise.
        """
        pass

    @abstractmethod
    def isfolder(self, name) -> bool:
        """Checks if given name is a folder in this storage.

        Args:
            name (str): Name of the entry to check

        Returns:
            bool: Returns True if entry is a folder, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, name) -> bool:
        """Deletes a given document.

        Args:
            name (str): File name.


        Returns:
            bool: Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def substorage(self, name: str) -> "Storage":
        """Returns a substorage, storage object with root in a subfolder.

        Args:
            name (str): Name of the entry, must be a folder

        Returns:
            Storage: Storage object
        """
        pass

    @abstractmethod
    def copy(self, localfile: str, destination: str):
        """Copy a document to another location

        Args:
            localfile (str): Original location
            destination (str): New location
        """
        pass

class NullStorage(Storage):
    """An implementation of dummy storage that does not save anything
    """

    #docstr_coverage:inherited
    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        return Results(self)

    def __repr__(self) -> str:
        return "<Null storage: {}>".format(self._root)

    #docstr_coverage:inherited
    def write(self, name, binary=False):
        if binary:
            return open(os.devnull, "wb")
        else:
            return open(os.devnull, "w")

    #docstr_coverage:inherited
    def documents(self):
        return []

    #docstr_coverage:inherited
    def folders(self):
        return []

    #docstr_coverage:inherited
    def read(self, name, binary=False):
        return None

    #docstr_coverage:inherited
    def isdocument(self, name):
        return False

    #docstr_coverage:inherited
    def isfolder(self, name):
        return False

    #docstr_coverage:inherited
    def delete(self, name) -> bool:
        return False

    #docstr_coverage:inherited
    def substorage(self, name):
        return NullStorage()

    #docstr_coverage:inherited
    def copy(self, localfile, destination):
        return

class LocalStorage(Storage):
    """Storage backed by the local filesystem.
    """

    def __init__(self, root: str):
        self._root = root
        self._results = os.path.join(root, "results")

    def __repr__(self) -> str:
        return "<Local storage: {}>".format(self._root)

    @property
    def base(self) -> str:
        return self._root

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        storage = LocalStorage(os.path.join(self._results, tracker.reference, experiment.identifier, sequence.name))
        return Results(storage)

    def documents(self):
        return [name for name in os.listdir(self._root) if os.path.isfile(os.path.join(self._root, name))]

    def folders(self):
        return [name for name in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, name))]

    def write(self, name: str, binary: bool = False):
        full = os.path.join(self.base, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        if binary:
            return open(full, mode="wb")
        else:
            return open(full, mode="w", newline="")

    def read(self, name, binary=False):
        full = os.path.join(self.base, name)

        if binary:
            return open(full, mode="rb")
        else:
            return open(full, mode="r", newline="")

    def delete(self, name) -> bool:
        full = os.path.join(self.base, name)
        if os.path.isfile(full):
            os.unlink(full)
            return True
        return False

    def isdocument(self, name):
        return os.path.isfile(os.path.join(self._root, name))

    def isfolder(self, name):
        return os.path.isdir(os.path.join(self._root, name))

    def substorage(self, name):
        return LocalStorage(os.path.join(self.base, name))

    def copy(self, localfile, destination):
        import shutil
        if os.path.isabs(destination):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, destination)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        shutil.move(localfile, os.path.join(self.base, full))

    def directory(self, *args):
        segments = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, str):
                segments.append(arg)
            elif isinstance(arg, (int, float)):
                segments.append(str(arg))
            else:
                segments.append(class_fullname(arg))

        path = os.path.join(self._root, *segments)
        os.makedirs(path, exist_ok=True)

        return path

class Cache(cachetools.Cache):
    """Persistent cache, extends the cache from cachetools package by storing cached objects (using picke serialization) to
    the underlying storage. 
    """

    def __init__(self, storage: Storage):
        """Creates a new cache backed by the given storage.

        Args:
            storage (Storage): The storage used to save objects.
        """
        super().__init__(10000)
        self._storage = storage

    def _filename(self, key: typing.Union[typing.Tuple, str]) -> str:
        """Generates a filename for the given object key.

        Args:
            key (typing.Union[typing.Tuple, str]): Cache key, either tuple or a single string

        Returns:
            str: Relative path as a string
        """
        if isinstance(key, tuple):
            filename = key[-1]
            if len(key) > 1:
                directory = self._storage.directory(*key[:-1])
            else:
                directory = ""
        else:
            filename = str(key)
            directory = ""
        return os.path.join(directory, filename)

    def __getitem__(self, key: str) -> typing.Any:
        """Retrieves an image from cache. If it does not exist, a KeyError is raised

        Args:
            key (str): Key of the item

        Raises:
            KeyError: Entry does not exist or cannot be retrieved
            PickleError: Unable to 

        Returns:
            typing.Any: item value
        """
        try:
            return super().__getitem__(key)
        except KeyError as e:
            filename = self._filename(key)
            if not self._storage.isdocument(filename):
                raise e
            try:
                with self._storage.read(filename, binary=True) as filehandle:
                    data = pickle.load(filehandle)
                    super().__setitem__(key, data)
                    return data
            except pickle.PickleError as e:
                raise KeyError(e)

    def __setitem__(self, key: str, value: typing.Any) -> None:
        """Sets an item for given key

        Args:
            key (str): Item key
            value (typing.Any): Item value

        """
        super().__setitem__(key, value)

        filename = self._filename(key)
        try:
            with self._storage.write(filename, binary=True) as filehandle:
                pickle.dump(value, filehandle)
        except pickle.PickleError:
            pass

    def __delitem__(self, key: str) -> None:
        """Operator for item deletion.

        Args:
            key (str): Key of object to remove
        """
        try:
            super().__delitem__(key)
            filename = self._filename(key)
            try:
                self._storage.delete(filename)
            except IOError:
                pass
        except KeyError:
            pass

    def __contains__(self, key: str) -> bool:
        """Magic method, does the cache include an item for a given key.

        Args:
            key (str): Item key

        Returns:
            bool: True if object exists for a given key
        """
        filename = self._filename(key)
        return self._storage.isdocument(filename)
