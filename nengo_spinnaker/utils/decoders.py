"""Tools for regularising the building and compression of decoders.
"""
import collections
import numpy as np


FunctionSolverEvals = collections.namedtuple(
    'FunctionSolverEvals', ['function', 'solver', 'eval_points'])


class DecoderBuilder(object):
    """Maintains a list of the decoders which have been built and responds to
    requests for new decoders with reference to this list.
    """
    def __init__(self, builder_function):
        """Create a new DecoderBuilder with the given function for building new
        decoders.

        The builder_function is expected to accept the function, eval_points
        and solver to use when solving for the decoder.
        """
        self.decoder_builder = builder_function
        self.built_decoders = dict()

    def get_transformed_decoder(self, function, transform,
                                eval_points, solver):
        """Return a transformed copy of the decoder for the given function,
        transform, eval_points and solver.
        """
        for cons, decoder in self.built_decoders.items():
            if (cons.function == function and cons.solver == solver and
                    np.all(cons.eval_points == eval_points)):
                break
        else:
            decoder = self.decoder_builder(function, eval_points, solver)
            key = FunctionSolverEvals(function, solver, eval_points)
            self.built_decoders[key] = decoder
        return np.dot(decoder, transform.T)
