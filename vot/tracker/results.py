"""Results module for storing and retrieving tracker results."""

import os
import fnmatch
from typing import List
from copy import copy
from vot.region import Region, Special, calculate_overlap, is_special
from vot.region.io import write_trajectory, read_trajectory
from vot.utilities import to_string

class Results(object):
    """Generic results interface for storing and retrieving results."""

    def __init__(self, storage: "vot.workspace.Storage"):
        """Creates a new results interface.

        :param storage: Storage interface
        :type storage: Storage
        """
        self._storage = storage

    def exists(self, name):
        """Returns true if the given file exists in the results storage.

        :param name: File name
        :type name: str

        :returns: True if the file exists
        :rtype: bool"""
        return self._storage.isdocument(name)

    def read(self, name):
        """Returns a file handle for reading the given file from the results storage.

        :param name: File name
        :type name: str

        :returns: File handle
        :rtype: file"""
        if name.endswith(".bin"):
            return self._storage.read(name, binary=True)
        return self._storage.read(name)

    def write(self, name: str):
        """Returns a file handle for writing the given file to the results storage.

        :param name: File name
        :type name: str

        :returns: File handle
        :rtype: file"""
        if name.endswith(".bin"):
            return self._storage.write(name, binary=True)
        return self._storage.write(name)

    def find(self, pattern):
        """Returns a list of files matching the given pattern in the results storage.

        :param pattern: Pattern
        :type pattern: str

        :returns: List of files
        :rtype: list"""

        return fnmatch.filter(self._storage.documents(), pattern)
    
class Trajectory(object):
    """Trajectory class for storing and retrieving tracker trajectories."""

    UNKNOWN = 0
    INITIALIZATION = 1
    FAILURE = 2

    @classmethod
    def exists(cls, results: Results, name: str) -> bool:
        """Returns true if the trajectory exists in the results storage.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str

        :returns: True if the trajectory exists
        :rtype: bool"""
        return results.exists(name + ".bin") or results.exists(name + ".txt")

    @classmethod
    def gather(cls, results: Results, name: str) -> list:
        """Returns a list of files that are part of the trajectory.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str

        :returns: List of files
        :rtype: list"""

        if results.exists(name + ".bin"):
            files = [name + ".bin"]
        elif results.exists(name + ".txt"):
            files = [name + ".txt"]
        else:
            return []

        for propertyfile in results.find(name + "_*.value"):
            files.append(propertyfile)

        return files

    @classmethod
    def read(cls, results: Results, name: str) -> 'Trajectory':
        """Reads a trajectory from the results storage.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str

        :returns: Trajectory
        :rtype: Trajectory"""

        def parse_float(line):
            """Parses a float from a line.

            :param line: Line
            :type line: str

            :returns: Float value
            :rtype: float"""
            if not line.strip():
                return None
            return float(line.strip())

        if results.exists(name + ".txt"):
            with results.read(name + ".txt") as fp:
                regions = read_trajectory(fp)
        elif results.exists(name + ".bin"):
            with results.read(name + ".bin") as fp:
                regions = read_trajectory(fp)
        else:
            raise FileNotFoundError("Trajectory data not found: {}".format(name))

        trajectory = Trajectory(len(regions))
        trajectory._regions = regions

        for propertyfile in results.find(name + "*.value"):
            with results.read(propertyfile) as filehandle:
                propertyname = os.path.splitext(os.path.basename(propertyfile))[0][len(name)+1:]
                lines = list(filehandle.readlines())
                try:
                    trajectory._properties[propertyname] = [parse_float(line) for line in lines]
                except ValueError:
                    trajectory._properties[propertyname] = [line.strip() for line in lines]

        return trajectory

    def __init__(self, length: int):
        """Creates a new trajectory of the given length.

        :param length: Trajectory length
        :type length: int
        """
        self._regions = [Special(Trajectory.UNKNOWN)] * length
        self._properties = dict()

    def set(self, frame: int, region: Region, properties: dict = None):
        """Sets the region for the given frame.

        :param frame: Frame index
        :type frame: int
        :param region: Region
        :type region: Region
        :param properties: Frame properties. Defaults to None.
        :type properties: dict, optional

        :raises IndexError: Frame index out of bounds"""
        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")

        self._regions[frame] = region

        if properties is None:
            properties = dict()

        for k, v in properties.items():
            if not k in self._properties:
                self._properties[k] = [None] * len(self._regions)
            self._properties[k][frame] = v

    def region(self, frame: int) -> Region:
        """Returns the region for the given frame.

        :param frame: Frame index
        :type frame: int

        :raises IndexError: Frame index out of bounds
        :returns: Region
        :rtype: Region"""
        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")
        return self._regions[frame]

    def regions(self) -> List[Region]:
        """Returns the list of regions.

        :returns: List of regions
        :rtype: List[Region]"""
        return copy(self._regions)

    def properties(self, frame: int = None) -> dict:
        """Returns the properties for the given frame or all properties if frame is
        None.

        :param frame: Frame index. Defaults to None.
        :type frame: int, optional

        :raises IndexError: Frame index out of bounds
        :returns: Properties
        :rtype: dict"""

        if frame is None:
            return tuple(self._properties.keys())

        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")

        return {k : v[frame] for k, v in self._properties.items() if not v[frame] is None}

    def __len__(self):
        """Returns the length of the trajectory.

        :returns: Length
        :rtype: int"""
        return len(self._regions)
    
    def __iter__(self):
        """Returns an iterator over the regions.

        :returns: Iterator
        :rtype: Iterator"""
        return iter(self._regions)

    def write(self, results: Results, name: str):
        """Writes the trajectory to the results storage.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str
        """
        from vot import config

        if config.results_binary:
            with results.write(name + ".bin") as fp:
                write_trajectory(fp, self._regions)
        else:
            with results.write(name + ".txt") as fp:
                # write_trajectory_file(fp, self._regions)
                write_trajectory(fp, self._regions)

        for k, v in self._properties.items():
            with results.write(name + "_" + k + ".value") as fp:
                fp.writelines([to_string(e) + "\n" for e in v])


    def equals(self, trajectory: 'Trajectory', check_properties: bool = False, overlap_threshold: float = 0.99999):
        """Returns true if the trajectories are equal.

        :param trajectory: _description_
        :type trajectory: Trajectory
        :param check_properties: _description_. Defaults to False.
        :type check_properties: bool, optional
        :param overlap_threshold: _description_. Defaults to 0.99999.
        :type overlap_threshold: float, optional

        :returns: _description_
        :rtype: _type_"""
        if not len(self) == len(trajectory):
            return False

        for r1, r2 in zip(self.regions(), trajectory.regions()):
            if calculate_overlap(r1, r2) < overlap_threshold and not (is_special(r1) and is_special(r2)):
                return False

        if check_properties:
            if not set(self._properties.keys()) == set(trajectory._properties.keys()):
                return False
            for name, _ in self._properties.items():
                for p1, p2 in zip(self._properties[name], trajectory._properties[name]):
                    if not p1 == p2 and not (p1 is None and p2 is None):
                        return False
        return True
