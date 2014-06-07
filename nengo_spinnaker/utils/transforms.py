"""Utilities for generating lists of unique transforms.
"""
import collections
import numpy as np

from nengo.utils.builder import full_transform


TransformCollection = collections.namedtuple(
    'TransformCollection', ['width', 'transforms_functions', 'connection_ids'])

TransformFunctionPair = collections.namedtuple(
    'TransformFunctionPair', ['transform', 'function'])

TransformFunctionWithSolver = collections.namedtuple(
    'TransformFunctionWithSolver', ['transform', 'function', 'solver'])


def get_transforms(connections):
    """Return a collection of unique transform/function pairs, the total width
    of this set and the index of each connection into the transform/function
    pair list.
    """
    transforms_functions = list()
    conn_indices = dict()

    for conn in connections:
        t = TransformFunctionPair(full_transform(conn, allow_scalars=False),
                                  conn.function)
        index = None

        for (i, t_) in enumerate(transforms_functions):
            if (np.all(t_.transform == t.transform) and
                    t_.function == t.function):
                index = i
                break
        else:
            transforms_functions.append(t)
            index = len(transforms_functions) - 1

        conn_indices[conn] = index

    width = sum([transform.transform.shape[0] for transform in
                 transforms_functions])

    return TransformCollection(width, transforms_functions, conn_indices)


def get_transforms_with_solvers(connections):
    """Return a collection of unique transform/function/solver triples, the
    total width of this set and the index of each connection into the
    transform/function/solver triples list.
    """
    transforms = get_transforms(connections)

    id_connections = collections.defaultdict(list)
    for (connection, index) in transforms.connection_ids.items():
        id_connections[index].append(connection)

    transforms_functions_solvers = list()
    conn_indices = dict()

    for (index, connections) in id_connections.items():
        solver_indices = dict()

        for connection in connections:
            if connection.solver not in solver_indices:
                transforms_functions_solvers.append(
                    TransformFunctionWithSolver(
                        transforms.transforms_functions[index].transform,
                        transforms.transforms_functions[index].function,
                        connection.solver
                    )
                )
                solver_indices[connection.solver] = (
                    len(transforms_functions_solvers) - 1)

            conn_indices[connection] = solver_indices[connection.solver]

    width = sum([transform.transform.shape[0] for transform in
                 transforms_functions_solvers])

    return TransformCollection(width, transforms_functions_solvers,
                               conn_indices)
