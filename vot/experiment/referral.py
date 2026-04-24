

from hashlib import md5

from attributee import String

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory, ObjectQuery
from vot.dataset import Sequence
from vot.region import Special

class ReferralExperiment(Experiment):
    """Experiment with text referral initialization.

    The experiment does not provide an initalization region, but a text prompt that
    describes the object to be tracked. The tracker is then expected to use this prompt
    to find the object in the sequence.
    """

    prompt_name = String(default="prompts")

    @property
    def _multiobject(self) -> bool:
        """Prevent SingleObject transformer from splitting sequences with ignore objects."""
        return True

    def _extract_prompt(self, sequence: Sequence):
        prompts = sequence.metadata(self.prompt_name, "")
        if not prompts:
            raise ValueError(f"Sequence {sequence.name} does not contain any prompts.")
        return prompts.split(";")

    def scan(self, tracker: Tracker, sequence: Sequence):
        """Scan the results of the experiment for the given tracker and sequence.

        :param tracker: The tracker to be scanned.
        :type tracker: Tracker
        :param sequence: The sequence to be scanned.
        :type sequence: Sequence

        :returns: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object.
        :rtype: [tuple]"""
        
        results = self.results(tracker, sequence)

        files = []
        complete = True
        assert len(sequence.objects()) == 1, "Referral experiment only supports single object sequences."
        
        prompts = self._extract_prompt(sequence)

        for prompt in prompts:
            prompt_hash = md5(prompt.encode()).hexdigest()[:8]
            name = f"{sequence.name}_{prompt_hash}"
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False
                break

        return complete, files, results

    def gather(self, tracker: Tracker, sequence: Sequence, objects = None, pad = False):
        """Gather trajectories for the given tracker and sequence.

        :param tracker: The tracker to be used.
        :type tracker: Tracker
        :param sequence: The sequence to be used.
        :type sequence: Sequence
        :param objects: The list of objects to be gathered. Defaults to None.
        :type objects: list, optional
        :param pad: Whether to pad the list of trajectories with None values. Defaults to False.
        :type pad: bool, optional

        :returns: The list of trajectories.
        :rtype: list"""
        trajectories = list()

        assert len(sequence.objects()) == 1, "Referral experiment only supports single object sequences."
        
        prompts = self._extract_prompt(sequence)

        results = self.results(tracker, sequence)
        
        for prompt in prompts:
            prompt_hash = md5(prompt.encode()).hexdigest()[:8]
            name = f"{sequence.name}_{prompt_hash}"
            if Trajectory.exists(results, name):
                trajectories.append(Trajectory.read(results, name))
            elif pad:
                trajectories.append(None)
        return trajectories
    
    def execute(self, tracker, sequence: Sequence, force = False, callback = None):
        
        prompts = self._extract_prompt(sequence)
        
        assert len(sequence.objects()) == 1, "Referral experiment only supports single object sequences."
        
        with self._get_runtime(tracker, sequence) as runtime:
        
            for i, prompt in enumerate(prompts):
                
                # Check if trajectory already exists
                prompt_hash = md5(prompt.encode()).hexdigest()[:8]
                name = f"{sequence.name}_{prompt_hash}"
                if Trajectory.exists(runtime.results(), name) and not force:
                    continue
                
                queries = [ObjectQuery(Special(0), {"prompt": prompt}, 0)]
                status = runtime.run(sequence, queries)

                trajectory = Trajectory(len(sequence))
                for i in range(len(sequence)):
                    region = status[i][0].region
                    trajectory.set(i, region)

                name = f"{sequence.name}_{prompt_hash}"
                Trajectory.write(runtime.results(), name, trajectory)

                if callback is not None:
                    callback(i / len(prompts))