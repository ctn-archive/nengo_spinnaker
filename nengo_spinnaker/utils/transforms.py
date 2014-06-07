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

TransformFunctionWithSolverEvalPoints = collections.namedtuple(
    'TransformFunctionWithSolverEvalPoints',
    ['transform', 'function', 'solver', 'eval_points'])


def get_transforms(connections):
    """Return a collection of unique transform/function pairs, the total width
    of this set and the index of each connection into the transform/function
    pair list.
    """
    transforms_functions = list()
    conn_indices = dict()

    # Ensure that all connections originate from the same place
    assert(np.all([connections[0].pre == c.pre for c in connections[1:]]))

    # For each connection generate a TransformFunctionPair, see if this pair
    # already exists.  If it does then record the index of the pair so that
    # the connection maps to it, otherwise add the TransformFunctionPair to the
    # list and record the new index.
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

    # The width of the collection is the accumulated size_outs of the
    # unique transform/function pairs.
    width = sum([transform.transform.shape[0] for transform in
                 transforms_functions])

    return TransformCollection(width, transforms_functions, conn_indices)


def get_transforms_with_solvers(connections):
    """Return a collection of unique transform/function/solver triples, the
    total width of this set and the index of each connection into the
    transform/function/solver triples list.
    """
    # Get a collection of unique transform/function pairs
    transforms = get_transforms(connections)

    # Build a dictionary mapping the index of a unique transform/function pairs
    # to a list of connections using that transform/function pair.
    id_connections = collections.defaultdict(list)
    for (connection, index) in transforms.connection_ids.items():
        id_connections[index].append(connection)

    transforms_functions_solvers = list()
    conn_indices = dict()

    # For each unique transform/function pair and its associated connections
    # build a list of unique solvers.  Where a solver is used by multiple
    # connections they may share the unique transform/function/solver triple,
    # otherwise a new transform/function/solver triple is added to the list.
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


def get_transforms_with_solvers_and_evals(connections):
    """Get a list of unique transform/function/solver/eval_points 4-tuples and
    and a dictionary mapping connections to indices of this list.
    """
    # Get a collection of unique transform/function/solver triples
    tfs = get_transforms_with_solvers(connections)

    id_connections = collections.defaultdict(list)
    for (connection, index) in tfs.connection_ids.items():
        id_connections[index].append(connection)

    transforms_functions_solvers_eps = list()
    conn_indices = dict()

    # For each indexed transform/function/solver triple and its associated
    # connections add a new transform/fnction/solver/eval_point 4-tuple for
    # each previously unseen collection of evaluation points.
    for (index, connections) in id_connections.items():
        eps = list()

        for connection in connections:
            j = None
            for (i, e) in enumerate(eps):
                if np.all(e == connection.eval_points):
                    j = i
                    break
            else:
                eps.append(connection.eval_points)
                transforms_functions_solvers_eps.append(
                    TransformFunctionWithSolverEvalPoints(
                        tfs.transforms_functions[index].transform,
                        tfs.transforms_functions[index].function,
                        tfs.transforms_functions[index].solver,
                        connection.eval_points
                    )
                )
                j = len(transforms_functions_solvers_eps) - 1

            conn_indices[connection] = j

    width = sum([transform.transform.shape[0] for transform in
                 transforms_functions_solvers_eps])

    return TransformCollection(width, transforms_functions_solvers_eps,
                               conn_indices)
