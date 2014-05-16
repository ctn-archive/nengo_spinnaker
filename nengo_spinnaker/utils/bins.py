"""Object collections for the Nengo/SpiNNaker Integration
"""

import itertools
import numpy as np

import nengo.decoders


class DecoderBinEntry(object):
    def __init__(self, decoder, func, transform=None):
        self.decoder = decoder
        self.func = func
        self.trans = transform


class DecoderBin(object):
    """A bin for decoders."""
    def __init__(self, rng):
        self._decoders = list()
        self._decoders_by_func = dict()
        self._decoders_edges = dict()
        self.rng = rng

    @property
    def decoder_widths(self):
        """Get the width of each decoder."""
        return [decoder.decoder.shape[1] for decoder in self._decoders]

    @property
    def width(self):
        return sum(self.decoder_widths)

    def get_decoder_index(self, e):
        """Get the decoder index for a given edge.

        ..todo::
          - Check function equivalency by comparing sampled points?
          - Check transform equivalency by comparing matrix values rather
              than entire matrix (e.g., dimension reduction can use the same
              transform despite being a different matrix.)
        """
        # Check if the decoder already exists for this function and transform
        for (i, dec) in enumerate(self._decoders):
            if dec.func == e.function and dec.trans == e.transform:
                self._decoders_edges[e] = i
                return i

        # Check if the decoder already exists for this function
        decoder = self._decoders_by_func.get(e.function, None)
        if decoder is None:
            # Compute the decoder
            eval_points = e.eval_points
            if eval_points is None:
                eval_points = e.prevertex.eval_points

            x = np.dot(eval_points, e.prevertex.encoders.T / e.pre.radius)
            activities = e.pre.neurons.rates(
                x, e.prevertex.gain, e.prevertex.bias
            )

            if e.function is None:
                targets = eval_points
            else:
                targets = np.array(
                    [e.function(ep) for ep in eval_points]
                )
                if targets.ndim < 2:
                    targets.shape = targets.shape[0], 1

            solver = e.decoder_solver
            if solver is None:
                solver = nengo.decoders.lstsq_L2nz

            decoder = solver(activities, targets, self.rng)

            if isinstance(decoder, tuple):
                decoder = decoder[0]

            self._decoders_by_func[e.function] = decoder

        # Combine the decoder with the transform and record it in the list
        decoder = np.dot(decoder, np.asarray(e.transform).T)
        self._decoders.append(
            DecoderBinEntry(decoder, e.transform, e.function)
        )
        i = len(self._decoders) - 1
        self._decoders_edges[e] = i

        return i

    def get_merged_decoders(self):
        """Get a merged decoder for the bin."""
        return np.hstack([d.decoder for d in self._decoders])

    def edge_index(self, edge):
        """Get the index of the decoder which this edge utilises."""
        return self._decoders_edges[edge]
