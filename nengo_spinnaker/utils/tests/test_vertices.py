"""Tests for Vertex functionality.
"""

import pytest

from nengo_spinnaker.utils import vertices


def test_ordered_regions():
    # Test that we get ordered regions
    r = vertices.ordered_regions('A', 'B', 'C', 'D', **{'E': 15})
    assert(r['A'] == 1)
    assert(r['B'] == 2)
    assert(r['C'] == 3)
    assert(r['D'] == 4)
    assert(r['E'] == 15)

    # Test that repeated regions throw an assertion error
    with pytest.raises(AssertionError):
        vertices.ordered_regions(**{'A': 1, 'B': 1})
