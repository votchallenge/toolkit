""" Dummy tracker for testing purposes. """

from __future__ import absolute_import
import os
from sys import path
import time

def _main_trax():
    """ Dummy TraX tracker main function for testing purposes."""
    from trax import Image, Region, Server, TraxStatus

    objects = None
    with Server([Region.RECTANGLE], [Image.PATH]) as server:
        while True:
            request = server.wait()
            if request.type in [TraxStatus.QUIT, TraxStatus.ERROR]:
                break
            if request.type == TraxStatus.INITIALIZE:
                objects = request.objects
            server.status(objects)
            time.sleep(0.1)

def _main_folder():
    """ Dummy tracker main function for testing purposes."""
   
    from vot.region.io import parse_region
   
    # Use the folder protocol, specified in vot.tracker.folder
   
    # List all files fllowing the pattern frames_*.txt in the current folder
    frames = [f for f in os.listdir() if f.startswith("frames_") and f.endswith(".txt")]
   
    # List all files following the pattern query_*.txt in the current folder
    queries = [f for f in os.listdir() if f.startswith("query_") and f.endswith(".txt")]
   
    assert len(frames) > 0, "No frames file found"
    assert len(queries) > 0, "No query file found"
   
    # Read image list from specified file
    with open(frames[0], "r", encoding="utf-8") as f:
        image_list = [line.strip() for line in f if line.strip()]

    # Read object queries from specified files
    object_queries = []
    for query_file in queries:
        with open(query_file, "r", encoding="utf-8") as f:
            object_id = query_file[len("query_"):-len(".txt")]
            lines = [line.strip() for line in f if line.strip()]
            offset = int(lines[0])
            state = parse_region(lines[1])
            properties = {}
            for line in lines[2:]:
                if "=" in line:
                    key, value = line.split("=", 1)
                    properties[key] = value
            object_queries.append((object_id, offset, state, properties))

    # Simulate tracking by writing output files for each object query
    for object_id, offset, state, properties in object_queries:
        with open(f"output_{object_id}.txt", "w", encoding="utf-8") as f:
            for j in range(len(image_list)):
                if j < offset:
                    f.write("0\n")
                else:
                    f.write(f"{state}\n")
        

if __name__ == '__main__':
    
    mode = os.environ.get("VOT_MODE", "trax")
    if mode == "trax":
        _main_trax()
    elif mode == "folder":
        _main_folder()
    else:
        raise RuntimeError(f"Unsupported VOT_MODE: {mode}")
else:
    from . import Tracker

    DummyTraxTracker = Tracker("dummy", __file__, "vot.tracker.dummy", "traxpython", paths=[], env_VOT_MODE="trax")
    DummyFolderTracker = Tracker("dummy", __file__, "vot.tracker.dummy", "folderpython", paths=[], env_VOT_MODE="folder")

