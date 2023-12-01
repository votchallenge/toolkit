

from typing import List

from attributee import Boolean

from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment.multirun import MultiRunExperiment, Experiment
from vot.report import ObjectVideo, SeparableReport


class PreviewVideos(SeparableReport):
    """A report that generates video previews for the tracker results."""

    groundtruth = Boolean(default=False, description="If set, the groundtruth is shown with the tracker output.")
    
    async def perexperiment(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):

        videos = []

        for sequence in sequences:

            for tracker in trackers:

                video = ObjectVideo(sequence.identifier + "_" + tracker.identifier, sequence)

                if self.groundtruth:
                    for frame in range(len(sequence)):
                        video(frame, "_", sequence.groundtruth(frame))

                for obj in sequence.objects():
                    trajectories = experiment.gather(tracker, sequence, objects=[obj])

                    if len(trajectories) == 0:
                        continue

                    for frame in range(len(sequence)):
                        video(frame, obj, trajectories[0].region(frame))
                
            videos.append(video)

        return videos

    def compatible(self, experiment):
        return isinstance(experiment, MultiRunExperiment)