from __future__ import absolute_import
import os
from sys import path
import time

from trax import Image, Region, Server, TraxStatus

def _main():
    region = None
    with Server([Region.RECTANGLE], [Image.PATH]) as server:
        while True:
            request = server.wait()
            if request.type in [TraxStatus.QUIT, TraxStatus.ERROR]:
                break
            if request.type == TraxStatus.INITIALIZE:
                region = request.region
            server.status(region)
            time.sleep(0.1)

if __name__ == '__main__':
    _main()
else:
    from . import Tracker

    DummyTracker = Tracker("dummy", __file__, "vot.tracker.dummy", "traxpython", paths=[])

