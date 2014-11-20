import nengo
import pytest

from ..import config


def test_f_of_t_parameter():
    """Check that the parameters for functions of time are sane.
    """
    model = nengo.Network()
    with model:
        m = nengo.Node(lambda t, x: 2*t, size_in=1)
        n = nengo.Node(lambda t: 2*t)

        # The following is illegal, so only Nodes with no incoming
        # connections could possibly be marked as functions of time.
        # >>> nengo.Connection(m, n)

    conf = config.Config()
    with pytest.raises(AttributeError):
        # m cannot be a function of time because it is also a function of x
        conf[m].f_of_t = True

    with pytest.raises(ValueError):
        # 7 is not a valid bool
        conf[n].f_of_t = 7
