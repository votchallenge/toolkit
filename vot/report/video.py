

from typing import List

from attributee import Boolean

from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment.multirun import MultiRunExperiment, Experiment
from vot.report import ObjectVideo, SeparableReport

class VideoWriter:

    def __init__(self, filename: str, fps: int = 30):
        self._filename = filename
        self._fps = fps

    def __call__(self, frame):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

class VideoWriterScikitH264(VideoWriter):

    def _handle(self):
        try:
            import skvideo.io
        except ImportError:
            raise ImportError("The scikit-video package is required for video export.")
        if not hasattr(self, "_writer"):
            skvideo.io.vwrite(self._filename, [])
            self._writer = skvideo.io.FFmpegWriter(self._filename, inputdict={'-r': str(self._fps), '-vcodec': 'libx264'})
        return self._writer

    def __call__(self, frame):
        self._handle().writeFrame(frame)

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.close()
            self._writer = None

class VideoWriterOpenCV(VideoWriter):


    def __init__(self, filename: str, fps: int = 30, codec: str = "mp4v"):
        super().__init__(filename, fps)
        self._codec = codec

    def __call__(self, frame):
        try:
            import cv2
        except ImportError:
            raise ImportError("The OpenCV package is required for video export.")
        if not hasattr(self, "_writer"):
            self._height, self._width = frame.shape[:2]
            self._writer = cv2.VideoWriter(self._filename, cv2.VideoWriter_fourcc(*self._codec.lower()), self._fps, (self._width, self._height))
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.release()
            self._writer = None

class PreviewVideos(SeparableReport):
    """A report that generates video previews for the tracker results."""

    groundtruth = Boolean(default=False, description="If set, the groundtruth is shown with the tracker output.")
    separate = Boolean(default=False, description="If set, each tracker is shown in a separate video.")

    async def perexperiment(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):

        videos = []

        for sequence in sequences:

            if self.separate:

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
            else:
                    
                video = ObjectVideo(sequence.identifier, sequence)

                if self.groundtruth:
                    for frame in range(len(sequence)):
                        video(frame, "_", sequence.groundtruth(frame))

                for tracker in trackers:

                    for obj in sequence.objects():
                        trajectories = experiment.gather(tracker, sequence, objects=[obj])

                        if len(trajectories) == 0:
                            continue

                        for frame in range(len(sequence)):
                            video(frame, obj + "_" + tracker.identifier, trajectories[0].region(frame))
                
                videos.append(video)

        return videos

    def compatible(self, experiment):
        return isinstance(experiment, MultiRunExperiment)