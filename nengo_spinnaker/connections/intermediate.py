import nengo
import numpy as np

from .reduced import (
    OutgoingReducedConnection, OutgoingReducedEnsembleConnection,
    IncomingReducedConnection, Target, _filter_types,
)

from ..ensemble.placeholder import PlaceholderEnsemble


class IntermediateConnection(object):
    """Intermediate representation of a connection object.
    """
    _expected_ensemble_type = (nengo.Ensemble, PlaceholderEnsemble)

    def __init__(self, pre_obj, post_obj, pre_slice=slice(None),
                 post_slice=slice(None), synapse=None, function=None,
                 transform=1., solver=None, eval_points=None, keyspace=None,
                 is_accumulatory=True, learning_rule_type=None):
        # Pre and post objects and slices
        self.pre_obj = pre_obj
        self.pre_slice = pre_slice
        self.post_obj = post_obj
        self.post_slice = post_slice

        # Other parameters
        self.transform = np.array(transform)
        self.synapse = synapse
        self.function = function
        self.solver = solver
        self.eval_points = eval_points
        self.keyspace = keyspace
        self.width = post_obj.size_in
        self.is_accumulatory = is_accumulatory
        self.learning_rule_type = learning_rule_type

    @classmethod
    def from_connection(cls, c, keyspace=None, is_accumulatory=True):
        """Return an IntermediateConnection object for any connections which
        have not already been replaced.  A requirement of any replaced
        connection type is that it has the attribute keyspace and can have
        its pre_obj and post_obj amended by later functions.
        """
        if not isinstance(c, nengo.Connection):
            raise NotImplementedError(
                "Cannot create a {} for an object of type `{}`.".format(
                    cls.__name__, c.__class__.__name__)
            )

        # Return a copy of this connection but with less information and
        # the full transform.
        return cls(c.pre_obj, c.post_obj, c.pre_slice, c.post_slice,
                   synapse=c.synapse, function=c.function,
                   transform=c.transform, solver=c.solver,
                   eval_points=c.eval_points, keyspace=keyspace,
                   learning_rule_type=c.learning_rule_type,
                   is_accumulatory=is_accumulatory)

    def __repr__(self):  # pragma: no cover
        return '<IntermediateConnection({}, {})>'.format(self.pre_obj,
                                                         self.post_obj)

    def _get_eval_points(self):
        # Try to get the eval points
        eval_points = self.eval_points
        if eval_points is None:
            assert self.pre_obj.eval_points is not None
            eval_points = np.array(self.pre_obj.eval_points)
        assert self.pre_obj.size_out == eval_points.shape[1]
        return eval_points

    def get_reduced_outgoing_connection(self):
        """Convert the IntermediateConnection into a reduced representation.
        """
        # Generate the outgoing component, depending on the nature of the
        # originating object.
        if not isinstance(self.pre_obj, self._expected_ensemble_type):
            return OutgoingReducedConnection(
                self.width, self.transform, self.function, self.pre_slice,
                self.post_slice, self.keyspace)
        else:
            if self.learning_rule_type is not None:
                # Check the learning rule affects transmission and include it.
                raise NotImplementedError

            # Get the eval points
            eval_points = self._get_eval_points()

            return OutgoingReducedEnsembleConnection(
                self.width, self.transform, self.function, self.pre_slice,
                self.post_slice, self.keyspace, eval_points, self.solver)

    def get_reduced_incoming_connection(self):
        """Convert the IntermediateConnection into a reduced representation.
        """
        return IncomingReducedConnection(
            Target(self.post_obj, self.post_slice), self._get_filter())

    def _get_filter(self):
        try:
            return _filter_types[self.synapse.__class__].from_synapse(
                self.synapse, self.is_accumulatory)
        except KeyError:
            raise NotImplementedError(self.synapse.__class__)
