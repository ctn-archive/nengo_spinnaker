import numpy as np


class DecoderBinEntry(object):
    def __init__(self, func, transform=None):
        self.func = func
        self.trans = transform


class DecoderBin(object):
    """A bin for decoders."""
    def __init__(self):
        self._decoders = list()
        self._decoders_by_func= dict()

    def get_decoder_index(self, c):
        """Get the index for a specific decoder."""
        # Check if the decoder already exists for this function and transform
        for (i,dec) in enumerate(self._decoders):
            if dec.func == c.function and dec.trans == conn.transform:
                return i

        # Check if the decoder already exists for this function
        decoder = self._decoders_by_func.get(c.function, None)
        if decoder is None:
            # Compute the decoder
            eval_points = c.eval_points
            if eval_points is None:
                raise NotImplementedError
            activities = c.pre.activities(eval_points)

            if c.function is None:
                targets = eval_points
            else:
                targets = np.array(
                    [c.function(ep) for ep in eval_points]
                )
                if targets.ndim < 2:
                    targets.shape = targets.shape[0], 1
            decoder = c.decoder_solver(activities, targets, self.rng)
            self.decoders_by_func[c.function] = decoder

        # Combine the decoder with the transform and record it in the list
        decoder = np.dot(decoder, c.transform.T)
        self._decoders.append((decoder, c.transform, c.function))

        return len(self._decoders) - 1

    def get_merged_decoders(self):
        """Get a merged decoder for the bin."""
        raise NotImplementedError
