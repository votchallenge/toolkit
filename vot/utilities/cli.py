import os
import sys
import argparse
import traceback
import logging
import yaml
from datetime import datetime
import colorama

from vot import check_updates, check_debug, __version__
from vot.tracker import Registry, TrackerException
from vot.stack import resolve_stack, list_integrated_stacks
from vot.workspace import Workspace, Cache
from vot.utilities import Progress, normalize_path, ColoredFormatter

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
    from vot.dataset.vot import VOTSequence
    trackers = Registry(config.registry)

    if not config.tracker:
        logger.error("Unable to continue without a tracker")
        logger.error("List of found trackers: ")
        for k in trackers.identifiers():
            logger.error(" * %s", k)
        return

    if not config.tracker in trackers:
        logger.error("Tracker does not exist")
        return

    tracker = trackers[config.tracker]

    logger.info("Generating dummy sequence")

    if config.sequence is None:
        sequence = DummySequence(50)
    else:
        sequence = VOTSequence(normalize_path(config.sequence))

    logger.info("Obtaining runtime for tracker %s", tracker.identifier)

    if config.visualize:
        import matplotlib.pylab as plt
        from vot.utilities.draw import MatplotlibDrawHandle
        figure = plt.figure()
        figure.canvas.set_window_title('VOT Test')
        axes = figure.add_subplot(1, 1, 1)
        axes.set_aspect("equal")
        handle = MatplotlibDrawHandle(axes, size=sequence.size)
        handle.style(fill=False)
        figure.show()

    runtime = None

    try:

        runtime = tracker.runtime(log=True)

        for repeat in range(1, 4):

            logger.info("Initializing tracker ({}/{})".format(repeat, 3))

            region, _, _ = runtime.initialize(sequence.frame(0), sequence.groundtruth(0))

            if config.visualize:
                axes.clear()
                handle.image(sequence.frame(0).channel())
                handle.style(color="green").region(sequence.frame(0).groundtruth())
                handle.style(color="red").region(region)
                figure.canvas.draw()

            for i in range(1, sequence.length):
                logger.info("Updating on frame %d/%d", i, sequence.length-1)
                region, _, _ = runtime.update(sequence.frame(i))

                if config.visualize:
                    axes.clear()
                    handle.image(sequence.frame(i).channel())
                    handle.style(color="green").region(sequence.frame(i).groundtruth())
                    handle.style(color="red").region(region)
                    figure.canvas.draw()

            logger.info("Stopping tracker")

        runtime.stop()

        logger.info("Test concluded successfuly")

    except TrackerException as te:
        logger.error("Error during tracker execution: {}".format(te))
        if runtime:
            runtime.stop()
    except KeyboardInterrupt:
        if runtime:
            runtime.stop()

def do_workspace(config, logger):

    from vot.workspace import WorkspaceException
    print(os.path.join(config.workspace, "configuration.m"))
    if config.stack is None and os.path.isfile(os.path.join(config.workspace, "configuration.m")):
        from vot.utilities.migration import migrate_matlab_workspace
        migrate_matlab_workspace(config.workspace)
        return
    elif config.stack is None:
        stacks = list_integrated_stacks()
        logger.error("Unable to continue without a stack")
        logger.error("List of available integrated stacks: ")
        for k, v in stacks.items():
            logger.error(" * %s - %s", k, v)

        return

    stack_file = resolve_stack(config.stack)

    if stack_file is None:
        logger.error("Experiment stack %s not found", stack_file)
        return

    default_config = dict(stack=config.stack, registry=["./trackers.ini"])

    try:
        Workspace.initialize(config.workspace, default_config, download=not config.nodownload)
        logger.info("Initialized workspace in '%s'", config.workspace)
    except WorkspaceException as we:
        logger.error("Error during workspace initialization: %s", we)

def do_evaluate(config, logger):

    from vot.experiment import run_experiment

    workspace = Workspace.load(config.workspace)

    logger.info("Loaded workspace in '%s'", config.workspace)

    global_registry = [os.path.abspath(x) for x in config.registry]

    registry = Registry(workspace.registry + global_registry, root=config.workspace)

    logger.info("Found data for %d trackers", len(registry))

    trackers = registry.resolve(*config.trackers, storage=workspace.storage.substorage("results"), skip_unknown=False)

    if len(trackers) == 0:
        logger.error("Unable to continue without at least on tracker")
        logger.error("List of available found trackers: ")
        for k in registry.identifiers():
            logger.error(" * %s", k)
        return

    try:
        for tracker in trackers:
            logger.info("Evaluating tracker %s", tracker.identifier)
            for experiment in workspace.stack:
                run_experiment(experiment, tracker, workspace.dataset, config.force, config.persist)

        logger.info("Evaluation concluded successfuly")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by the user")
    except TrackerException as te:
        logger.error("Evaluation interrupted by tracker error: {}".format(te))

def do_analysis(config, logger):

    from vot.analysis.processor import AnalysisProcessor, process_stack_analyses
    from vot.document import generate_document

    workspace = Workspace.load(config.workspace)

    logger.info("Loaded workspace in '%s'", config.workspace)

    global_registry = [os.path.abspath(x) for x in config.registry]

    registry = Registry(workspace.registry + global_registry, root=config.workspace)

    logger.info("Found data for %d trackers", len(registry))

    if not config.trackers:
        trackers = workspace.list_results(registry)
    else:
        trackers = registry.resolve(*config.trackers, storage=workspace.storage.substorage("results"), skip_unknown=False)

    if not trackers:
        logger.warning("No trackers resolved, stopping.")
        return

    logger.debug("Running analysis for %d trackers", len(trackers))

    if config.workers == 1:
        from vot.utilities import ThreadPoolExecutor
        executor = ThreadPoolExecutor(1)

    else:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(config.workers)

    if config.nocache:
        from cachetools import LRUCache
        cache = LRUCache(1000)
    else:
        cache = Cache(workspace.cache("analysis"))

    with AnalysisProcessor(executor, cache):

        results = process_stack_analyses(workspace, trackers)

        if results is None:
            return

        if config.name is None:
            name = "{:%Y-%m-%dT%H-%M-%S.%f%z}".format(datetime.now())
        else:
            name = config.name

        storage = workspace.storage.substorage("analysis").substorage(name)

        generate_document(config.format, workspace.report, trackers, workspace.dataset, results, storage)

    executor.shutdown(wait=True)

    logger.info("Analysis successful, report available as %s", name)


def do_pack(config, logger):

    import zipfile, io
    from shutil import copyfileobj

    workspace = Workspace.load(config.workspace)

    logger.info("Loaded workspace in '%s'", config.workspace)

    registry = Registry(workspace.registry + config.registry, root=config.workspace)

    logger.info("Found data for %d trackers", len(registry))

    tracker = registry[config.tracker]

    logger.info("Packaging results for tracker %s", tracker.identifier)

    all_files = []
    can_finish = True

    with Progress("Scanning", len(workspace.dataset) * len(workspace.stack)) as progress:

        for experiment in workspace.stack:
            for sequence in workspace.dataset:
                sequence = experiment.transform(sequence)
                complete, files, results = experiment.scan(tracker, sequence)
                all_files.extend([(f, experiment.identifier, sequence.name, results) for f in files])
                if not complete:
                    logger.error("Results are not complete for experiment %s, sequence %s", experiment.identifier, sequence.name)
                    can_finish = False
                progress.relative(1)

    if not can_finish:
        logger.error("Unable to continue, experiments not complete")
        return

    logger.info("Collected %d files, compressing to archive ...", len(all_files))

    timestamp = datetime.now()

    archive_name = "{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.zip".format(tracker.identifier, timestamp)

    with Progress("Compressing", len(all_files)) as progress:

        manifest = dict(identifier=tracker.identifier, configuration=tracker.configuration(),
            timestamp="{:%Y-%m-%dT%H-%M-%S.%f%z}".format(timestamp), platform=sys.platform,
            python=sys.version, toolkit=__version__)

        with zipfile.ZipFile(workspace.storage.write(archive_name, binary=True)) as archive:
            for f in all_files:
                info = zipfile.ZipInfo(filename=os.path.join(f[1], f[2], f[0]), date_time=timestamp.timetuple())
                with io.TextIOWrapper(archive.open(info, mode="w")) as fout, f[3].read(f[0]) as fin:
                    copyfileobj(fin, fout)
                progress.relative(1)

            info = zipfile.ZipInfo(filename="manifest.yml", date_time=timestamp.timetuple())
            with io.TextIOWrapper(archive.open(info, mode="w")) as fout:
                yaml.dump(manifest, fout)

    logger.info("Result packaging successful, archive available in %s", archive_name)

def main():
    logger = logging.getLogger("vot")
    stream = logging.StreamHandler()
    stream.setFormatter(ColoredFormatter())
    logger.addHandler(stream)

    parser = argparse.ArgumentParser(description='VOT Toolkit Command Line Utility', prog="vot")
    parser.add_argument("--debug", "-d", default=False, help="Backup backend", required=False, action='store_true')
    parser.add_argument("--registry", default=".", help='Tracker registry paths', required=False, action=EnvDefault, \
        separator=os.path.pathsep, envvar='VOT_REGISTRY')

    subparsers = parser.add_subparsers(help='commands', dest='action', title="Commands")

    test_parser = subparsers.add_parser('test', help='Test a tracker integration on a synthetic sequence')
    test_parser.add_argument("tracker", help='Tracker identifier', nargs="?")
    test_parser.add_argument("--visualize", "-g", default=False, required=False, help='Visualize results of the test session', action='store_true')
    test_parser.add_argument("--sequence", "-s", required=False, help='Path to sequence to use instead of dummy')

    workspace_parser = subparsers.add_parser('initialize', help='Setup a new workspace and download data')
    workspace_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    workspace_parser.add_argument("--nodownload", default=False, required=False, help="Do not download dataset if specified in stack", action='store_true')
    workspace_parser.add_argument("stack", nargs="?", help='Experiment stack')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate one or more trackers in a given workspace')
    evaluate_parser.add_argument("trackers", nargs='+', default=None, help='Tracker identifiers')
    evaluate_parser.add_argument("--force", "-f", default=False, help="Force rerun of the entire evaluation", required=False, action='store_true')
    evaluate_parser.add_argument("--persist", "-p", default=False, help="Persist execution even in case of an error", required=False, action='store_true')
    evaluate_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')

    analysis_parser = subparsers.add_parser('analysis', help='Run analysis of results')
    analysis_parser.add_argument("trackers", nargs='*', help='Tracker identifiers')
    analysis_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    analysis_parser.add_argument("--format", choices=("html", "latex", "pdf", "json", "yaml"), default="html", help='Analysis output format')
    analysis_parser.add_argument("--name", required=False, help='Analysis output name')
    analysis_parser.add_argument("--workers", default=1, required=False, help='Number of parallel workers', type=int)
    analysis_parser.add_argument("--nocache", default=False, required=False, help="Do not cache data to disk", action='store_true')

    pack_parser = subparsers.add_parser('pack', help='Package results for submission')
    pack_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    pack_parser.add_argument("tracker", help='Tracker identifier')

    try:

        args = parser.parse_args()

        logger.setLevel(logging.INFO)

        if args.debug or check_debug:
            logger.setLevel(logging.DEBUG)

        update, version = check_updates()
        if update:
            logger.warning("A newer version of VOT toolkit is available (%s), please update.", version)

        if args.action == "test":
            do_test(args, logger)
        elif args.action == "initialize":
            do_workspace(args, logger)
        elif args.action == "evaluate":
            do_evaluate(args, logger)
        elif args.action == "analysis":
            do_analysis(args, logger)
        elif args.action == "pack":
            do_pack(args, logger)
        else:
            parser.print_help()

    except argparse.ArgumentError as e:
        logger.error(e)
        exit(-1)
    except Exception as e:
        logger.exception(e)
        exit(1)

    exit(0)