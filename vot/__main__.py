
import os
import argparse
import traceback

from vot.tracker import load_trackers

class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if not default and envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

def do_test(config):
    from vot.dataset.dummy import DummySequence
    trackers = load_trackers(config.registry)

    if not config.tracker in trackers:
        print("Tracker does not exist")
        return

    tracker = trackers[config.tracker]

    sequence = DummySequence()

    runtime = tracker.runtime()

    runtime.initialize(sequence.frame(0), sequence.groundtruth(0))

    for i in range(1, sequence.length-1):
        #print(i)
        runtime.update(sequence.frame(i))



def do_workspace(config):
    pass

def do_evaluate(config):
    
    from vot.workspace import Workspace

    registry = load_trackers(config.registry)
    workspace = Workspace(config.workspace)

    trackers = [registry[t.trim()] for t in config.trackers.split(",")]
    
    for tracker in trackers:
        for experiment in workspace.stack:
            for sequence in workspace.dataset:
                experiment.execute(tracker, sequence, experiment.results(tracker, experiment, sequence))


def do_analysis(config):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VOT Toolkit Command Line Utility', prog="vot")
    parser.add_argument("--debug", "-d", action=EnvDefault, envvar='VOT_DEBUG', default=False, help="Backup backend", required=False)
    parser.add_argument("--registry", default=".", help='Tracker registry path', required=False, action=EnvDefault, envvar='VOT_REGISTRY')
    #parser.add_argument("--database", default=".", help='Global sequence database', required=False)

    subparsers = parser.add_subparsers(help='commands', dest='action', title="Commands")

    test_parser = subparsers.add_parser('test', help='Test a tracker integration on a synthetic sequence')
    test_parser.add_argument("tracker", help='Tracker identifier')
    test_parser.add_argument("--visualize", "-g", default=False, help='Visualize results of the test session')

    workspace_parser = subparsers.add_parser('workspace', help='Setup a new workspace and download data')
    workspace_parser.add_argument("--workspace", default=".", help='Workspace path')
    workspace_parser.add_argument("stack", help='Experiment stack')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate one or more trackers in a given workspace')
    evaluate_parser.add_argument("trackers", nargs='?', default=None, help='Tracker identifiers')
    evaluate_parser.add_argument("--workspace", default=".", help='Workspace path')

    analysis_parser = subparsers.add_parser('analysis', help='Run interactive analysis')
    analysis_parser.add_argument("--workspace", default=".", help='Workspace path')

    try:

        args = parser.parse_args()

        if args.action == "test":
            do_test(args)
        elif args.action == "workspace":
            do_workspace(args)
        elif args.action == "evaluate":
            do_evaluate(args)
        elif args.action == "analyze":
            do_analysis(args)
        else:
            parser.print_help()

    except argparse.ArgumentError:
        traceback.print_exc()


    exit(0)