"""Uniform tasks, then uniform samples; deterministic continuation by draw index."""
import random
from collections import defaultdict


class TaskBalancedSampler:
    def __init__(self, samples, draws, seed=0, start=0):
        self.groups = defaultdict(list)
        for index, row in enumerate(samples):
            self.groups[row["task"]].append(index)
        self.tasks = sorted(self.groups)
        if not self.tasks or draws < 1 or not 0 <= start < draws:
            raise ValueError("Invalid sampler data/budget/start")
        self.draws, self.seed, self.start = draws, seed, start

    def __len__(self):
        return self.draws - self.start

    def __iter__(self):
        generator = random.Random(self.seed)
        for draw in range(self.draws):
            task = generator.choice(self.tasks)
            index = generator.choice(self.groups[task])
            if draw >= self.start:
                yield index
