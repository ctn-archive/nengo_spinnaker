"""Tests for function of time Node types.
"""

import numpy as np

from pacman.model.graph_mapper.slice import Slice
from .. import f_of_t
from ...utils.fixpoint import bitsk


def test_compress_transform():
    """Check that transforms can be reduced correctly."""
    transform = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    new_t, indices = f_of_t.get_compressed_transform(transform)

    assert indices == [0, 2, 3]
    assert np.all(new_t == np.array([[1, 0], [1, 1], [0, 1]]))


def test_create_data_region():
    """Check that the data region for a function of time Node is constructed
    correctly.  We don't really care what type of region it is, just that the
    size and formatted data is appropriate.
    """
    # Construct the sample function
    fn = lambda t: np.array([np.sin(2*np.pi*t), np.cos(2*np.pi*t)])

    # Construct the transforms and functions
    transform_functions = [
        f_of_t.FunctionAndTransform(None, np.eye(2)),
        f_of_t.FunctionAndTransform(
            lambda x: x**2, np.array([[0.5, 0], [0, 0.5], [0.5, 0.5]]))
    ]

    # Get the evaluation points
    eval_points = np.arange(1000) * 0.001

    # Construct the reference data
    evals = [np.array(fn(t)) for t in eval_points]
    reference_data = list()
    for e in evals:
        data = list()
        for tf in transform_functions:
            _e = tf.function(e) if tf.function is not None else e
            data.extend(bitsk(np.dot(tf.transform, _e).tolist()))
        reference_data.append(data)

    reference_data = np.array(reference_data, dtype=np.uint32)
    assert reference_data.shape == (eval_points.size, 5)

    # Create the data region, there are 5 atoms
    data_region = f_of_t.create_data_region(fn, transform_functions,
                                            eval_points)
    assert not data_region.in_dtcm

    # Check that the sizing is correct, for some samples
    assert data_region.sizeof(Slice(0, 0)) == eval_points.size
    assert data_region.sizeof(Slice(2, 4)) == 3 * eval_points.size
    assert data_region.sizeof(Slice(1, 2)) == 2 * eval_points.size

    # Check that the eventual data is correct
    for s in (Slice(0, 0), Slice(2, 4), Slice(1, 2)):
        sr = data_region.create_subregion(s, 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        sample_data = reference_data.T[s.as_slice].T.flatten()

        assert np.all(sample_data == sr_data)
