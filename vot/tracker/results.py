
"""Results module for storing and retrieving tracker results."""

import os
import fnmatch
from typing import List
from copy import copy
from vot.region import Region, RegionType, Special, calculate_overlap
from vot.region.io import write_trajectory, read_trajectory
from vot.utilities import to_string

class Results(object):
    """Generic results interface for storing and retrieving results."""

    def __init__(self, storage: "Storage"):
        """Creates a new results interface.
        
        Args:
            storage (Storage): Storage interface
        """
        self._storage = storage

    def exists(self, name):
        """Returns true if the given file exists in the results storage.

        Args:
            name (str): File name

        Returns:
            bool: True if the file exists
        """
        return self._storage.isdocument(name)

    def read(self, name):
        """Returns a file handle for reading the given file from the results storage.

        Args:
            name (str): File name

        Returns:
            file: File handle
        """
        if name.endswith(".bin"):
            return self._storage.read(name, binary=True)
        return self._storage.read(name)

    def write(self, name: str):
        """Returns a file handle for writing the given file to the results storage.

        Args:
            name (str): File name
        
        Returns:
            file: File handle
        """
        if name.endswith(".bin"):
            return self._storage.write(name, binary=True)
        return self._storage.write(name)

    def find(self, pattern):
        """Returns a list of files matching the given pattern in the results storage.

        Args:
            pattern (str): Pattern

        Returns:
            list: List of files
        """

        return fnmatch.filter(self._storage.documents(), pattern)
    
class Trajectory(object):
    """Trajectory class for storing and retrieving tracker trajectories."""

    UNKNOWN = 0
    INITIALIZATION = 1
    FAILURE = 2

    @classmethod
    def exists(cls, results: Results, name: str) -> bool:
        """Returns true if the trajectory exists in the results storage.
        
        Args:
            results (Results): Results storage
            name (str): Trajectory name (without extension)
            
        Returns:
            bool: True if the trajectory exists
        """
        return results.exists(name + ".bin") or results.exists(name + ".txt")

    @classmethod
    def gather(cls, results: Results, name: str) -> list:
        """Returns a list of files that are part of the trajectory.
        
        Args:
            results (Results): Results storage
            name (str): Trajectory name (without extension)
            
        Returns:
            list: List of files
        """

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
        
        Args:
            results (Results): Results storage
            name (str): Trajectory name (without extension)
            
        Returns:
            Trajectory: Trajectory
        """

        def parse_float(line):
            """Parses a float from a line.
            
            Args:
                line (str): Line
                
            Returns:
                float: Float value
            """
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

        Args:
            length (int): Trajectory length
        """
        self._regions = [Special(Trajectory.UNKNOWN)] * length
        self._properties = dict()

    def set(self, frame: int, region: Region, properties: dict = None):
        """Sets the region for the given frame.

        Args:
            frame (int): Frame index
            region (Region): Region
            properties (dict, optional): Frame properties. Defaults to None.

        Raises:
            IndexError: Frame index out of bounds
        """
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

        Args:
            frame (int): Frame index

        Raises:
            IndexError: Frame index out of bounds

        Returns:
            Region: Region
        """
        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")
        return self._regions[frame]

    def regions(self) -> List[Region]:
        """ Returns the list of regions. 
        
        Returns:
            List[Region]: List of regions
        """
        return copy(self._regions)

    def properties(self, frame: int = None) -> dict:
        """Returns the properties for the given frame or all properties if frame is None.

        Args:
            frame (int, optional): Frame index. Defaults to None.

        Raises:
            IndexError: Frame index out of bounds

        Returns:
            dict: Properties
        """

        if frame is None:
            return tuple(self._properties.keys())

        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")

        return {k : v[frame] for k, v in self._properties.items() if not v[frame] is None}

    def __len__(self):
        """Returns the length of the trajectory.

        Returns:
            int: Length
        """
        return len(self._regions)

    def write(self, results: Results, name: str):
        """Writes the trajectory to the results storage. 

        Args:
            results (Results): Results storage
            name (str): Trajectory name (without extension)
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

        Args:
            trajectory (Trajectory): _description_
            check_properties (bool, optional): _description_. Defaults to False.
            overlap_threshold (float, optional): _description_. Defaults to 0.99999.

        Returns:
            _type_: _description_
        """
        if not len(self) == len(trajectory):
            return False

        for r1, r2 in zip(self.regions(), trajectory.regions()):
            if calculate_overlap(r1, r2) < overlap_threshold and not (r1.type == RegionType.SPECIAL and r2.type == RegionType.SPECIAL):
                return False

        if check_properties:
            if not set(self._properties.keys()) == set(trajectory._properties.keys()):
                return False
            for name, _ in self._properties.items():
                for p1, p2 in zip(self._properties[name], trajectory._properties[name]):
                    if not p1 == p2 and not (p1 is None and p2 is None):
                        return False
        return True
