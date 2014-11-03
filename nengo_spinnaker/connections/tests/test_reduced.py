import copy
import mock
import numpy as np

from ..reduced import (
    OutgoingReducedConnection, OutgoingReducedEnsembleConnection,
    IncomingReducedConnection, Target, LowpassFilterParameter,
    StandardInputPort, GlobalInhibitionPort,
)


def assert_equal_not_equiv(a, b):
    assert a == b
    assert a is not b


def comparison_ids(items):
    return ((i, j) for i in range(len(items)) for j in range(i+1, len(items)))


class TestOutgoingReducedConnection(object):
    def test_eq(self):
        # Create some reduced connections and test for equivalence
        rcs = [
            OutgoingReducedConnection(3, np.eye(3), None, slice(None),
                                      slice(None)),
            OutgoingReducedConnection(3, np.eye(3), None, slice(None),
                                      slice(None)),
            OutgoingReducedConnection(3, np.eye(3), lambda x: x**2,
                                      slice(None), slice(None)),
            OutgoingReducedConnection(1, np.eye(1), None,
                                      slice(None), slice(None)),
            OutgoingReducedConnection(3, np.eye(3), None, slice(None),
                                      slice(None), mock.Mock()),
            OutgoingReducedEnsembleConnection(3, np.eye(3), None, slice(None),
                                              slice(None)),
            OutgoingReducedConnection(3, np.eye(3), None, slice(0, 1),
                                      slice(None)),
            OutgoingReducedConnection(3, np.eye(3), None, slice(None),
                                      slice(0, 1)),
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

    def test_eq_with_slices(self):
        rcs = [
            OutgoingReducedConnection(3, np.eye(3), None, slice(0, 1),
                                      slice(1, 2)),
            OutgoingReducedConnection(3, np.eye(3), None, slice(0, 1),
                                      slice(1, 2)),
            OutgoingReducedConnection(3, np.eye(3), None,
                                      pre_slice=slice(1, 2),
                                      post_slice=slice(1, 2)),
            OutgoingReducedConnection(3, np.eye(3), None,
                                      pre_slice=slice(0, 1),
                                      post_slice=slice(0, 1)),
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
        c = OutgoingReducedConnection(
            3, np.eye(3), lambda x: x**2, slice(None), slice(None), mock.Mock()
        )
        d = copy.copy(c)

        # Assert the transforms are equivalent but not the same object
        assert np.all(d.transform == c.transform)
        assert c.transform is not d.transform, (id(c.transform),
                                                id(d.transform))
        assert_equal_not_equiv(c.pre_slice, d.pre_slice)
        assert_equal_not_equiv(c.post_slice, d.post_slice)

        # Assert that the function, width and keyspace are equivalent
        assert c.function is d.function
        assert c.width == d.width
        assert c.keyspace is d.keyspace


class TestOutgoingReducedEnsembleConnection(object):
    def test_eq_no_func_equiv(self):
        # Create some reduced connections and test for equivalence without
        # considering function equivalence.
        eval_points = np.linspace(-1., 1.)
        rcs = [
            OutgoingReducedEnsembleConnection(
                3, np.eye(3), None, slice(None), slice(None), None,
                eval_points),
            OutgoingReducedEnsembleConnection(
                3, np.eye(3), None, slice(None), slice(None), None,
                eval_points),
            OutgoingReducedEnsembleConnection(
                3, np.eye(3), None, slice(None), slice(None), None, None),
            OutgoingReducedEnsembleConnection(
                3, np.eye(3), None, slice(None), slice(None), None,
                eval_points, mock.Mock()),
            OutgoingReducedConnection(
                3, np.eye(3), None, slice(None), slice(None), None),
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
            OutgoingReducedEnsembleConnection(
                1, 1., lambda x: x**2, slice(None), slice(None), None,
                eval_points),
            OutgoingReducedEnsembleConnection(
                1, 1., lambda x: x**2, slice(None), slice(None), None,
                eval_points),
            OutgoingReducedEnsembleConnection(
                1, 1., lambda x: x**3, slice(None), slice(None), None,
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

    def test_copy(self):
        eval_points = np.linspace(-1., 1.)
        ks = mock.Mock()
        solver = mock.Mock()
        transmitter_learning_rules = (mock.Mock(),)

        rcs = OutgoingReducedEnsembleConnection(
            1, 2., lambda x: x**2, pre_slice=slice(None),
            post_slice=slice(None), keyspace=ks, eval_points=eval_points,
            solver=solver,
            transmitter_learning_rule=transmitter_learning_rules)

        rcs2 = copy.copy(rcs)

        # Assert that most parameters are copied across correctly
        assert rcs2 is not rcs
        assert rcs2.width == rcs.width
        assert rcs.function is rcs2.function
        assert rcs.keyspace is rcs2.keyspace
        assert rcs.solver is rcs2.solver
        assert rcs.transmitter_learning_rule == rcs2.transmitter_learning_rule

        # Assert that eval_points and transform are equivalent but NOT the same
        # object.
        assert np.all(rcs.transform == rcs2.transform)
        assert np.all(rcs.eval_points == rcs2.eval_points)
        assert rcs.transform is not rcs2.transform
        assert rcs.eval_points is not rcs2.eval_points

        # Check pre/post slices
        assert_equal_not_equiv(rcs.pre_slice, rcs2.pre_slice)
        assert_equal_not_equiv(rcs.post_slice, rcs2.post_slice)

    def test_eq_learning_rules(self):
        # Create some reduced connections and test for equivalence when
        # learning rules are present.
        lrule = mock.Mock()
        rcs = [
            OutgoingReducedEnsembleConnection(
                1, 1., None, slice(None), slice(None),
                transmitter_learning_rule=lrule),
            OutgoingReducedEnsembleConnection(
                1, 1., None, slice(None), slice(None),
                transmitter_learning_rule=lrule),
            OutgoingReducedEnsembleConnection(
                1, 1., None, slice(None), slice(None))
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
            IncomingReducedConnection(obj_a, None),
            IncomingReducedConnection(obj_b, None),
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

    def test_copy(self):
        obj_a = mock.Mock()
        target = Target(obj_a, slice(None))
        filter_object = LowpassFilterParameter(0.03)

        irc = IncomingReducedConnection(target, filter_object)
        new_irc = copy.copy(irc)

        # Assert that this is more like a deep copy
        assert_equal_not_equiv(irc.target, new_irc.target)
        assert_equal_not_equiv(irc.filter_object, new_irc.filter_object)


class TestTargetObject(object):
    """Test target/port objects.
    """
    def test_target_eqs(self):
        # Check target equivalences
        o1 = mock.Mock()
        p1 = StandardInputPort
        p2 = mock.Mock()
        o2 = GlobalInhibitionPort
        rcs = [
            Target(o1, slice(0, 1), p1),
            Target(o1, slice(0, 1), p1),
            Target(o1, slice(None), p1),
            Target(o1, slice(None), p2),
            Target(o2, slice(None), p1),
            Target(o2, slice(None), p2),
        ]

        eqs = [True] + [False]*(len(rcs) - 1)
        hashes = [True, False] + [False]*(len(rcs) - 2)

        i_s = ((i, j) for i in range(len(rcs)) for j in range(i+1, len(rcs)))
        for (i, j), eq, he in zip(i_s, eqs, hashes):
            assert (rcs[i] == rcs[j]) is eq, "[{}] {} [{}]".format(
                i, "==" if eq else "!=", j)
            assert (hash(rcs[i]) == hash(rcs[j])) is he, (i, j)

    def test_copy(self):
        obj_a = mock.Mock()

        t = Target(obj_a, slice(None))
        u = copy.copy(t)

        # Assert referents are the same but that the Targets themselves aren't.
        # This is the default behaviour of copy but it's necessary for correct
        # operation, so a unit test seems like a good idea.
        assert_equal_not_equiv(t, u)
        assert t.target_object is u.target_object
        assert t.port is u.port
        assert_equal_not_equiv(t.slice, u.slice)


class TestLowpassFilterParameter(object):
    def test_eq(self):
        filters = [
            LowpassFilterParameter(0.03),
            LowpassFilterParameter(0.02),
            LowpassFilterParameter(0.03, is_accumulatory=False),
        ]
        i_s = comparison_ids(filters)

        for (i, j) in i_s:
            assert filters[i] != filters[j], (i, j)
            assert hash(filters[i]) != hash(filters[j]), (i, j)

    def test_from_synapse(self):
        import nengo

        synapse = nengo.Lowpass(0.05)
        f = LowpassFilterParameter.from_synapse(synapse, False)
        assert f.tau == synapse.tau
        assert not f.is_accumulatory
