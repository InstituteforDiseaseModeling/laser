"""Base class for all models."""

from tqdm import tqdm


class DiseaseModel:
    """Base class for all disease models."""

    def __init__(self):
        return

    def initialize(self) -> None:
        raise NotImplementedError

    def step(self, tick: int, pbar: tqdm) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def run(self, ticks: int) -> None:
        for tick in (pbar := tqdm(range(ticks))):
            self.step(tick, pbar)
        return

    def serialize(self, filename: str) -> None:
        raise NotImplementedError

    @classmethod
    def deserialize(self, filename: str) -> "DiseaseModel":
        raise NotImplementedError
