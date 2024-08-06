import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import List, Text, Dict, Iterator


class Preloader(IterDataPipe):
    """Preload elements of the underlying dataset to generate in-memory examples.

    This transformer can lead to considerable performance improvements at the cost of
    memory consumption.

    This is functionally equivalent to `tf.data.Dataset.cache`, except the cached
    examples are accessible directly via the `examples` attribute.

    Attributes:
        source_dp: The input `IterDataPipe` with examples that contain `"images"` key.
        examples: Stored list of preloaded elements.

    """

    def __init__(self, source_dp: IterDataPipe):
        self.source_dp = source_dp
        self.examples = list(iter(source_dp))
        # self.cache: Dict[Any, T_co] = {}

    @property
    def input_keys(self) -> List[Text]:
        """Return the keys that incoming elements are expected to have."""
        return []

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys that outgoing elements will have."""
        return []

    # def __getitem__(self, index) -> T_co:
    #     if index not in self.cache:
    #         self.cache[index] = self.source_dp[index]  # type: ignore[index]
    #     return self.cache[index]  # type: ignore[index]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for ex in self.examples:
            yield ex
