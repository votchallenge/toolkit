import os
import argparse
import traceback
import logging
import json
from datetime import datetime

from vot.tracker import load_trackers
from vot.stack import resolve_stack
from vot.workspace import Workspace
from vot.utilities import Progress

class EvaluationProgress(object):

    def __init__(self, description, total):
        self.bar = Progress(desc=description, total=total, unit="sequence")
        self._finished = 0

    def __call__(self, progress):
        self.bar.update_absolute(self._finished + min(1, max(0, progress)))

    def push(self):
        self._finished = self._finished + 1
        self.bar.update_absolute(self._finished)

class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, separator=None, **kwargs):
        if not default and envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if separator:
            default = default.split(separator)
        if required and default:
            required = False
        self.separator = separator
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if self.separator:
            values = values.split(self.separator)
        setattr(namespace, self.dest, values)

def do_test(config, logger):
    from vot.dataset.dummy import DummySequence
    trackers = load_trackers(config.registry)

    if not config.tracker in trackers:
        logger.error("Tracker does not exist")
        return

    tracker = trackers[config.tracker]

    logger.info("Generating dummy sequence")

    sequence = DummySequence()

    logger.info("Obtaining runtime for tracker %s", tracker.identifier)

    runtime = tracker.runtime(log=True)

    logger.info("Initializing tracker")

    runtime.initialize(sequence.frame(0), sequence.groundtruth(0))

    for i in range(1, sequence.length-1):
        logger.info("Updating on frame %d/%d", i, sequence.length-1)
        runtime.update(sequence.frame(i))

    logger.info("Stopping tracker")

    runtime.stop()

    logger.info("Test concluded successfuly")

def do_workspace(config, logger):
    
    from vot.workspace import initialize_workspace, migrate_workspace

    if config.stack is None and os.path.isfile(os.path.join(config.workspace, "configuration.m")):
        migrate_workspace(config.workspace)
        return
    elif config.stack is None:
        logger.error("Unable to continue without a stack")
        return

    stack_file = resolve_stack(config.stack)

    if stack_file is None:
        logger.error("Experiment stack not found")
        return

    default_config = dict(stack=config.stack, registry=["trackers"])

    initialize_workspace(config.workspace, default_config)

    logger.info("Initialized workspace in '%s'", config.workspace)

def do_evaluate(config, logger):

    workspace = Workspace(config.workspace)

    logger.info("Loaded workspace in '%s'", config.workspace)

    global_registry = [os.path.abspath(x) for x in config.registry]

    registry = load_trackers(workspace.registry + global_registry, root=config.workspace)

    logger.info("Found data for %d trackers", len(registry))

    try:
        trackers = [registry[t.strip()] for t in config.trackers]
    except KeyError as ke:
        logger.error("Tracker not found %s", str(ke))
        return

    try:

        for tracker in trackers:
            logger.info("Evaluating tracker %s", tracker.identifier)
            for experiment in workspace.stack:
                progress = EvaluationProgress("{}/{}".format(tracker.identifier, experiment.identifier), len(workspace.dataset))
                for sequence in workspace.dataset:
                    experiment.execute(tracker, sequence, force=config.force, callback=progress)
                    progress.push()

        logger.info("Evaluation concluded successfuly")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by the user")

def do_analysis(config, logger):

    from vot.analysis import process_measures

    workspace = Workspace(config.workspace)

    logger.info("Loaded workspace in '%s'", config.workspace)

    registry = load_trackers(workspace.registry + config.registry)

    logger.info("Found data for %d trackers", len(registry))

    if not hasattr(config, 'trackers'):
        trackers = workspace.list_results()
    else:
        trackers = config.trackers

    try:
        trackers = [registry[tracker] for tracker in trackers]
    except KeyError as ke:
        logger.error("Tracker not found %s", str(ke))
        return

    if config.output == "dash":
        from vot.analysis.dashboard import run_dashboard
        run_dashboard(workspace, trackers)
        return
    elif config.output == "latex":
        pass
    elif config.output == "html":
        pass
    elif config.output == "json":
        results = process_measures(workspace, trackers)
        file_name = os.path.join(workspace.directory, "analysis_{:%Y-%m-%dT%H:%M:%S.%f%z}.json".format(datetime.now()))
        with open(file_name, "w") as fp:
            json.dump(results, fp)

    logger.info("Analysis successful, results available in %s", file_name)


def do_pack(config, logger):

    import zipfile, io
    from shutil import copyfileobj

    workspace = Workspace(config.workspace)

    logger.info("Loaded workspace in '%s'", config.workspace)

    registry = load_trackers(workspace.registry + config.registry)

    logger.info("Found data for %d trackers", len(registry))

    try:
        tracker = registry[config.tracker]
    except KeyError as ke:
        logger.error("Tracker not found %s", str(ke))
        return

    logger.info("Packaging results for tracker %s", tracker.identifier)

    all_files = []
    can_finish = True

    progress = Progress(desc="Scanning", total=len(workspace.dataset) * len(workspace.stack))

    for experiment in workspace.stack:
        for sequence in workspace.dataset:
            complete, files, results = experiment.scan(tracker, sequence)
            all_files.extend([(f, experiment.identifier, sequence.name, results) for f in files])
            if not complete:
                logger.error("Results are not complete for experiment %s, sequence %s", experiment.identifier, sequence.name) 
                can_finish = False
            progress.update_relative(1)

    if not can_finish:
        logger.error("Unable to continue, experiments not complete")
        return

    logger.info("Collected %d files, compressing to archive ...", len(all_files))

    archive_name = os.path.join(workspace.directory, "{}_{:%Y-%m-%dT%H:%M:%S.%f%z}.zip".format(tracker.identifier, datetime.now()))

    progress = Progress(desc="Compressing", total=len(all_files))

    with zipfile.ZipFile(archive_name, 'w') as archive:
        for f in all_files:
            with io.TextIOWrapper(archive.open(os.path.join(f[1], f[2], f[0]), mode="w")) as fout, f[3].read(f[0]) as fin:
                copyfileobj(fin, fout)
            progress.update_relative(1)

    logger.info("Result packaging successful, archive available in %s", archive_name)


def main():
    logger = logging.getLogger("vot")
    logger.addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser(description='VOT Toolkit Command Line Utility', prog="vot")
    parser.add_argument("--debug", "-d", default=False, help="Backup backend", required=False, action='store_true')
    parser.add_argument("--registry", default=".", help='Tracker registry paths', required=False, action=EnvDefault, \
        separator=os.path.pathsep, envvar='VOT_REGISTRY')

    subparsers = parser.add_subparsers(help='commands', dest='action', title="Commands")

    test_parser = subparsers.add_parser('test', help='Test a tracker integration on a synthetic sequence')
    test_parser.add_argument("tracker", help='Tracker identifier')
    test_parser.add_argument("--visualize", "-g", default=False, help='Visualize results of the test session')

    workspace_parser = subparsers.add_parser('workspace', help='Setup a new workspace and download data')
    workspace_parser.add_argument("--workspace", default=".", help='Workspace path')
    workspace_parser.add_argument("stack", nargs="?", help='Experiment stack')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate one or more trackers in a given workspace')
    evaluate_parser.add_argument("trackers", nargs='+', default=None, help='Tracker identifiers')
    evaluate_parser.add_argument("--force", "-f", default=False, help="Force rerun of the entire evaluation", required=False)
    evaluate_parser.add_argument("--workspace", default=".", help='Workspace path')

    analysis_parser = subparsers.add_parser('analysis', help='Run interactive analysis')
    analysis_parser.add_argument("trackers", nargs='*', help='Tracker identifiers')
    analysis_parser.add_argument("--workspace", default=".", help='Workspace path')
    analysis_parser.add_argument("--output", choices=("dash", "latex", "html", "json"), default="dash", help='Analysis output format')

    pack_parser = subparsers.add_parser('pack', help='Package results for submission')
    pack_parser.add_argument("--workspace", default=".", help='Workspace path')
    pack_parser.add_argument("tracker", help='Tracker identifier')

    try:

        args = parser.parse_args()

        if args.debug:
            logger.setLevel(logging.DEBUG)

        if args.action == "test":
            do_test(args, logger)
        elif args.action == "workspace":
            do_workspace(args, logger)
        elif args.action == "evaluate":
            do_evaluate(args, logger)
        elif args.action == "analysis":
            do_analysis(args, logger)
        elif args.action == "pack":
            do_pack(args, logger)
        else:
            parser.print_help()

    except argparse.ArgumentError:
        traceback.print_exc()

    exit(0)