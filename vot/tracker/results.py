
import os
import fnmatch
from typing import List
from copy import copy
from vot.region import Region, RegionType, Special, write_file, read_file, calculate_overlap
from vot.utilities import to_string

class Results(object):

    def __init__(self, storage: "Storage"):
        self._storage = storage

    def exists(self, name):
        return self._storage.isdocument(name)

    def read(self, name):
        return self._storage.read(name)

    def write(self, name):
        return self._storage.write(name)

    def find(self, pattern):
        return fnmatch.filter(self._storage.documents(), pattern)

class Trajectory(object):

    @classmethod
    def exists(cls, results: Results, name: str) -> bool:
        return results.exists(name + ".txt")

    @classmethod
    def gather(cls, results: Results, name: str) -> list:

        if not Trajectory.exists(results, name):
            return []

        files = [name + ".txt"]

        for propertyfile in results.find(name + "_*.value"):
            files.append(propertyfile)

        return files

    @classmethod
    def read(cls, results: Results, name: str) -> 'Trajectory':

        def parse_float(line):
            if not line.strip():
                return None
            return float(line.strip())

        if not results.exists(name + ".txt"):
            raise FileNotFoundError("Trajectory data not found")

        with results.read(name + ".txt") as fp:
            regions = read_file(fp)

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

    def __init__(self, length:int):
        self._regions = [Special(Special.UNKNOWN)] * length
        self._properties = dict()

    def set(self, frame: int, region: Region, properties: dict = None):
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
        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")
        return self._regions[frame]

    def regions(self) -> List[Region]:
        return copy(self._regions)

    def properties(self, frame: int = None) -> dict:

        if frame is None:
            return tuple(self._properties.keys())

        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")

        return {k : v[frame] for k, v in self._properties.items() if not v[frame] is None}

    def __len__(self):
        return len(self._regions)

    def write(self, results: Results, name: str):

        with results.write(name + ".txt") as fp:
            write_file(fp, self._regions)

        for k, v in self._properties.items():
            with results.write(name + "_" + k + ".value") as fp:
                fp.writelines([to_string(e) + "\n" for e in v])


    def equals(self, trajectory: 'Trajectory', check_properties: bool = False, overlap_threshold: float = 0.99999):
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
