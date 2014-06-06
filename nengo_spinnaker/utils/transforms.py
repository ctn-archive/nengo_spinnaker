"""Utilities for generating lists of unique transforms.
"""
import collections
import numpy as np

from nengo.builder.utils import full_transform


TransformCollection = collections.namedtuple(
    'TransformCollection', ['width', 'transforms', 'edge_indices'])


def get_transforms(vertex):
    """Return a tuple of width, unique transforms and edge indices."""
    transforms = list()
    edge_indices = dict()

    for edge in vertex.out_edges:
        t = full_transform(edge.conn, allow_scalars=False)

        if not t in transforms:
            transforms.append(t)
        edge_indices(edge) = transforms.index(t)

    width = sum([t.shape[0] for t in transforms])

    return TransformCollection(width, transforms, edge_indices)
