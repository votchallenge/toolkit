

"""Video report helpers and report element implementations."""

from typing import List

from attributee import Boolean

from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment.multirun import MultiRunExperiment, Experiment
from vot.report import ObjectVideo, SeparableReport

class VideoWriter:
    """Abstract interface for writing a stream of rendered frames."""

    def __init__(self, filename: str, fps: int = 30):
        """Initialize writer target and frame rate."""
        self._filename = filename
        self._fps = fps

    def __call__(self, frame):
        """Append a frame to the output stream."""
        raise NotImplementedError()

    def close(self):
        """Finalize and close underlying resources."""
        raise NotImplementedError()

class VideoWriterScikitH264(VideoWriter):
    """FFmpeg-backed H.264 writer implemented via scikit-video."""

    def _handle(self):
        """Create or return the underlying FFmpeg writer handle."""
        try:
            import skvideo.io
        except ImportError:
            raise ImportError("The scikit-video package is required for video export.")
        if not hasattr(self, "_writer"):
            skvideo.io.vwrite(self._filename, [])
            self._writer = skvideo.io.FFmpegWriter(self._filename, inputdict={'-r': str(self._fps), '-vcodec': 'libx264'})
        return self._writer

    def __call__(self, frame):
        """Write a single RGB frame to the H.264 stream."""
        self._handle().writeFrame(frame)

    def close(self):
        """Close the FFmpeg writer if it was initialized."""
        if hasattr(self, "_writer"):
            self._writer.close()
            self._writer = None

class VideoWriterOpenCV(VideoWriter):
    """Video writer that uses OpenCV codecs (typically AVI/MP4)."""


    def __init__(self, filename: str, fps: int = 30, codec: str = "mp4v"):
        """Initialize OpenCV writer with an explicit fourcc codec."""
        super().__init__(filename, fps)
        self._codec = codec

    def __call__(self, frame):
        """Append one RGB frame to the OpenCV stream."""
        try:
            import cv2
        except ImportError:
            raise ImportError("The OpenCV package is required for video export.")
        if not hasattr(self, "_writer"):
            self._height, self._width = frame.shape[:2]
            self._writer = cv2.VideoWriter(self._filename, cv2.VideoWriter_fourcc(*self._codec.lower()), self._fps, (self._width, self._height))
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        """Release the OpenCV writer if it was initialized."""
        if hasattr(self, "_writer"):
            self._writer.release()
            self._writer = None

class PreviewVideos(SeparableReport):
    """A report that generates video previews for the tracker results."""

    groundtruth = Boolean(default=False, description="If set, the groundtruth is shown with the tracker output.")
    separate = Boolean(default=False, description="If set, each tracker is shown in a separate video.")

    def _populate_video(self, video: ObjectVideo, experiment: Experiment, trackers: List[Tracker], sequence: Sequence):
        """Draw trajectories for all requested trackers and objects into a video."""

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

    async def generate_video(self, experiment: Experiment, trackers: List[Tracker], sequence: Sequence, identifier: str = None):
        """Build one preview video for a sequence and a set of trackers."""
        video = ObjectVideo(identifier or sequence.identifier, sequence)
        self._populate_video(video, experiment, trackers, sequence)
        
        return video

    async def perexperiment(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        """Generate per-sequence videos, optionally split per tracker."""

        videos = []

        for sequence in sequences:

            if self.separate:
                for tracker in trackers:
                    video = await self.generate_video(
                        experiment,
                        [tracker],
                        sequence,
                        identifier="{}_{}".format(sequence.identifier, tracker.identifier),
                    )
                    videos.append(video)
            else:
                video = await self.generate_video(experiment, trackers, sequence)
                videos.append(video)

        return videos

    def compatible(self, experiment):
        """Restrict this report element to multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)