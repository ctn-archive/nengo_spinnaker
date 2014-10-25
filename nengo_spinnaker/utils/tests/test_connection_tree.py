import copy
import mock
import numpy as np

from .. import connection_tree
from ..connection_tree import (
    OutgoingReducedConnection, OutgoingReducedEnsembleConnection,
    IncomingReducedConnection, IncomingReducedEnsembleConnection,
)


class TestOutgoingReducedConnection(object):
    def test_eq(self):
        # Create some reduced connections and test for equivalence
        rcs = [
            OutgoingReducedConnection(np.eye(3), None),
            OutgoingReducedConnection(np.eye(3), None),
            OutgoingReducedConnection(np.eye(3), lambda x: x**2),
            OutgoingReducedConnection(np.eye(1), None),
            OutgoingReducedConnection(np.eye(3), None, mock.Mock()),
            OutgoingReducedEnsembleConnection(np.eye(3), None),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_copy(self):
        # Ensure that when we copy we get copies of the transform but
        # references to everything else.
        c = connection_tree.OutgoingReducedConnection(
            np.eye(3), lambda x: x**2, mock.Mock())
        d = copy.copy(c)

        # Assert the transforms are equivalent but not the same object
        assert np.all(d.transform == c.transform)
        assert c.transform is not d.transform, (id(c.transform),
                                                id(d.transform))

        # Assert that the function and keyspace are equivalent
        assert c.function is d.function
        assert c.keyspace is d.keyspace

    def test_copy_with_transform(self):
        # Ensure that we can copy a reduced connection but modify the transform
        # in the process.
        c = connection_tree.OutgoingReducedConnection(np.eye(3), None)
        d = c.copy_with_transform(3.)

        assert np.all(d.transform == c.transform*3.)
        assert c.function is d.function
        assert c.keyspace is d.keyspace


class TestOutgoingReducedEnsembleConnection(object):
    def test_eq_no_func_equiv(self):
        # Create some reduced connections and test for equivalence without
        # considering function equivalence.
        eval_points = np.linspace(-1., 1.)
        rcs = [
            OutgoingReducedEnsembleConnection(np.eye(3), None, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(np.eye(3), None, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(np.eye(3), None, None, None),
            OutgoingReducedEnsembleConnection(np.eye(3), None, None,
                                              eval_points, mock.Mock()),
            OutgoingReducedConnection(np.eye(3), None, None),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_eq(self):
        # Create some reduced connections and test for equivalence
        eval_points = np.linspace(-1., 1.)
        rcs = [
            OutgoingReducedEnsembleConnection(1., lambda x: x**2, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(1., lambda x: x**2, None,
                                              eval_points),
            OutgoingReducedEnsembleConnection(1., lambda x: x**3, None,
                                              eval_points),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_eq_learning_rules(self):
        # Create some reduced connections and test for equivalence when
        # learning rules are present.
        lrules = [mock.Mock()]
        rcs = [
            OutgoingReducedEnsembleConnection(
                1., None, transmitter_learning_rules=lrules),
            OutgoingReducedEnsembleConnection(
                1., None, transmitter_learning_rules=lrules),
            OutgoingReducedEnsembleConnection(
                1., None, transmitter_learning_rules=[]),
        ]

        # Define what the equivalences should be
        eqs = [False]*(len(rcs))

        # Define whether the hashes should be equivalent
        hashes = [True] + eqs[1:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)


class TestIncomingReducedConnection(object):
    def test_eq(self):
        # Create some incoming reduced connections and test for equivalence.
        obj_a = mock.Mock()
        obj_b = mock.Mock()
        rcs = [
            IncomingReducedConnection(obj_a, None),
            IncomingReducedConnection(obj_a, None),  # Same target, same filter
            IncomingReducedConnection(obj_b, None),  # Diff target, same filter
            IncomingReducedConnection(obj_a, mock.Mock()),
            IncomingReducedConnection(obj_b, mock.Mock()),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)


class TestIncomingReducedEnsembleConnection(object):
    def test_eq(self):
        # Create some incoming reduced connections and test for equivalence.
        obj_a = mock.Mock()
        obj_b = mock.Mock()
        rcs = [
            IncomingReducedEnsembleConnection(obj_a, None),
            IncomingReducedEnsembleConnection(obj_a, None),
            IncomingReducedEnsembleConnection(obj_b, None),
            IncomingReducedEnsembleConnection(obj_a, mock.Mock()),
            IncomingReducedEnsembleConnection(obj_b, mock.Mock()),
        ]

        # Define what the equivalences should be
        eqs = [True] + [False]*(len(rcs) - 1)

        # Define whether the hashes should be equivalent
        hashes = eqs[:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_eq_with_learning(self):
        # Create some incoming reduced connections and test for equivalence.
        obj_a = mock.Mock()
        lrules = [mock.Mock()]
        rcs = [
            IncomingReducedEnsembleConnection(
                obj_a, None, receiver_learning_rules=lrules),
            IncomingReducedEnsembleConnection(
                obj_a, None, receiver_learning_rules=lrules),
            IncomingReducedEnsembleConnection(
                obj_a, None)
        ]

        # Define what the equivalences should be
        eqs = [False]*len(rcs)

        # Define whether the hashes should be equivalent
        hashes = [True] + eqs[1:]

        # Get the set of indices for equivalence
        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, (i, j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)
