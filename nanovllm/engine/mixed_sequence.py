from nanovllm.engine.sequence import Sequence


class MixedSequence(Sequence):
    """Keep BlockManager's field name; expose the actual computed-prefix meaning."""

    def __init__(self, token_ids, sampling_params):
        super().__init__(token_ids, sampling_params)
        self.computed_high_water = 0

    @property
    def num_computed_tokens(self):
        return self.num_cached_tokens

    # This object stays on rank 0. Workers receive BatchMetadata, not Sequence.
