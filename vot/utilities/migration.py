

import os
import re
import logging

import yaml
import numpy as np

from vot.tracker import is_valid_identifier
from vot.stack import resolve_stack
from vot.workspace import WorkspaceException

def migrate_matlab_workspace(directory):

    logger = logging.getLogger("vot")

    logger.info("Attempting to migrate workspace in %s", directory)

    def scan_text(pattern, content, default=None):
        matches = re.findall(pattern, content)
        if not len(matches) == 1:
            return default
        return matches[0]

    config_file = os.path.join(directory, "config.yaml")
    if os.path.isfile(config_file):
        raise WorkspaceException("Workspace already initialized")

    old_config_file = os.path.join(directory, "configuration.m")
    if not os.path.isfile(old_config_file):
        raise WorkspaceException("Old workspace config not detected")

    with open(old_config_file, "r") as fp:
        content = fp.read()
        stack = scan_text("set\\_global\\_variable\\('stack', '([A-Za-z0-9-_]+)'\\)", content)
        if stack is None:
            raise WorkspaceException("Experiment stack could not be retrieved")

    tracker_ids = list()

    for tracker_dir in [x for x in os.scandir(os.path.join(directory, "results")) if x.is_dir()]:
        if not is_valid_identifier(tracker_dir.name):
            logger.info("Results directory %s is not a valid identifier, skipping.", tracker_dir.name)
            continue
        logger.debug("Scanning results for %s", tracker_dir.name)
        tracker_ids.append(tracker_dir.name)
        for experiment_dir in [x for x in os.scandir(tracker_dir.path) if x.is_dir()]:
            for sequence_dir in [x for x in os.scandir(experiment_dir.path) if x.is_dir()]:
                timing_file = os.path.join(sequence_dir.path, "{}_time.txt".format(sequence_dir.name))
                if os.path.isfile(timing_file):
                    logger.debug("Migrating %s", timing_file)
                    times = np.genfromtxt(timing_file, delimiter=",")
                    if len(times.shape) == 1:
                        times = np.reshape(times, (times.shape[0], 1))
                    for k in range(times.shape[1]):
                        if np.all(times[:, k] == 0):
                            break
                        np.savetxt(os.path.join(sequence_dir.path, \
                             "%s_%03d_time.value" % (sequence_dir.name, k+1)), \
                             times[:, k] / 1000, fmt='%.6e')
                    os.unlink(timing_file)

    trackers = dict()

    for tid in tracker_ids:
        old_description = os.path.join(directory, "tracker_{}.m".format(tid))
        label = tid
        if os.path.isfile(old_description):
            with open(old_description, "r") as fp:
                content = fp.read()
                label = scan_text("tracker\\_label *= * ['\"](.*)['\"]", content, tid)
        trackers[tid] = dict(label=label, protocol="unknown", command="")

    if trackers:
        with open(os.path.join(directory, "trackers.ini"), "w") as fp:
            for tid, tdata in trackers.items():
                fp.write("[" + tid + "]\n")
                for k, v in tdata.items():
                    fp.write(k + " = " + v + "\n")
                fp.write("\n\n")

    if resolve_stack(stack) is None:
        logger.warning("Stack %s not found, you will have to manually edit and correct config file.", stack)

    with open(config_file, 'w') as fp:
        yaml.dump(dict(stack=stack, registry=["."]), fp)

    #os.unlink(old_config_file)

    logger.info("Workspace %s migrated", directory)