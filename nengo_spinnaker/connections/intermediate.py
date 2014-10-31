import nengo
import numpy as np

from .reduced import (
    OutgoingReducedConnection, OutgoingReducedEnsembleConnection,
    IncomingReducedConnection, Target, _filter_types,
)


class IntermediateConnection(object):
    """Intermediate representation of a connection object.
    """
    _expected_ensemble_type = nengo.Ensemble

    def __init__(self, pre_obj, post_obj, synapse=None, function=None,
                 transform=1., solver=None, eval_points=None, keyspace=None,
                 is_accumulatory=True, learning_rule=None):
        self.pre_obj = pre_obj
        self.post_obj = post_obj

        self.transform = np.array(transform)
        self.synapse = synapse
        self.function = function
        self.solver = solver
        self.eval_points = eval_points
        self.keyspace = keyspace
        self.width = post_obj.size_in
        self.is_accumulatory = is_accumulatory
        self.learning_rule = learning_rule

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

        # Get the full transform
        tr = nengo.utils.builder.full_transform(c, allow_scalars=False)

        # Return a copy of this connection but with less information and
        # the full transform.
        return cls(c.pre_obj, c.post_obj, synapse=c.synapse,
                   function=c.function, transform=tr, solver=c.solver,
                   eval_points=c.eval_points, keyspace=keyspace,
                   learning_rule=c.learning_rule,
                   is_accumulatory=is_accumulatory)

    def __repr__(self):  # pragma: no cover
        return '<IntermediateConnection({}, {})>'.format(self.pre_obj,
                                                         self.post_obj)

    def get_reduced_outgoing_connection(self):
        """Convert the IntermediateConnection into a reduced representation.
        """
        # Generate the outgoing component, depending on the nature of the
        # originating object.
        if not isinstance(self.pre_obj, self._expected_ensemble_type):
            return OutgoingReducedConnection(
                self.width, self.transform, self.function, self.keyspace)
        else:
            if self.learning_rule is not None:
                # Check the learning rule affects transmission and include it.
                raise NotImplementedError

            # Try to get the eval points
            eval_points = self.eval_points
            if eval_points is None:
                assert self.pre_obj.eval_points is not None
                eval_points = np.array(self.pre_obj.eval_points)
            assert self.pre_obj.size_out == eval_points.shape[1]

            return OutgoingReducedEnsembleConnection(
                self.width, self.transform, self.function, self.keyspace,
                eval_points, self.solver)

    def get_reduced_incoming_connection(self):
        """Convert the IntermediateConnection into a reduced representation.
        """
        return IncomingReducedConnection(Target(self.post_obj),
                                         self._get_filter())

    def _get_filter(self):
        try:
            return _filter_types[self.synapse.__class__].from_synapse(
                self.synapse, self.is_accumulatory)
        except KeyError:
            raise NotImplementedError(self.synapse.__class__)
