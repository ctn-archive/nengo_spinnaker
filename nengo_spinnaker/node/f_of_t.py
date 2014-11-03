import collections
import numpy as np

from ..utils.fixpoint import bitsk
from ..spinnaker import regions


FunctionAndTransform = collections.namedtuple('FunctionAndTransform',
                                              'function transform')


def get_compressed_transform(transform, threshold=0.0):
    """Compress a transform by removing redundant rows.
    """
    retain = np.any(transform > threshold, axis=1)
    new_transform = np.vstack(transform[retain])
    return new_transform, np.where(retain)[0].tolist()


def create_data_region(fn, transform_functions, eval_points):
    """Create a new region to represent the evaluation of the given function
    transformed as required for output.
    """
    # Determine what the shape should be
    shape = (len(eval_points), sum(tf.transform.shape[0] for tf in
                                   transform_functions))

    # Apply the function to all evaluation points and transform the output at
    # each point to ready it for output.
    transformed_values = np.zeros(shape)
    for i, t in enumerate(eval_points):
        v = fn(t)  # Get the value for this timestep

        j = 0
        for tf in transform_functions:
            jn = j + tf.transform.shape[0]
            v2 = tf.function(v) if tf.function is not None else v
            transformed_values[i][j:jn] = tf.transform.dot(v2)
            j = jn

    # Create the data region
    region = regions.MatrixRegionPartitionedByColumns(
        transformed_values, shape=shape, formatter=bitsk, in_dtcm=False)
    return region
