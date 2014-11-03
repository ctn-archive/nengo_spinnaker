import nengo
import numpy as np

from .. import probe as probe_tools


class TestInsertDecodedOutputProbes(object):
    def test_simple(self):
        # Create a new Network and then add probe objects
        with nengo.Network():
            a = nengo.Ensemble(100, 2)
            b = nengo.Node(lambda t, x: x, size_in=2, size_out=2)

            ps = [
                nengo.Probe(a.neurons, 'spikes'),
                nengo.Probe(a[0], 'decoded_output', synapse=0.03),
                nengo.Probe(a),
                nengo.Probe(b)
            ]

        # Insert decoded output probe objects
        objs, conns = probe_tools.insert_decoded_output_probes([], [], ps)

        # Assert list lengths are sensible
        assert len(objs) == 3
        assert len(conns) == 3

        # Check that probe objects are correct
        _u = list(ps)
        for o in objs:
            assert o.probe in _u
            p = _u.pop(_u.index(o.probe))
            assert o.sample_every == p.sample_every
            assert o.size_in == p.size_in

        # Check that probe connections are correct
        for c in conns:
            assert c.post_obj in objs

            if c.post_obj.probe is ps[1]:
                assert c.synapse.tau == 0.03
                assert c.pre_obj is a
                assert np.all(c.transform == 1.)
            elif c.post_obj.probe is ps[2]:
                assert c.pre_obj is a
                assert np.all(c.transform == np.eye(2)) or c.transform == 1.
            elif c.post_obj.probe is ps[3]:
                assert c.pre_obj is b
            else:
                assert False, "Unknown probe..."
