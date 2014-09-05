import collections


def get_profile_data(txrx, profiled_vertices):
    """Build a dictionary of object->vertices->subvertices->profile data.
    """
    profile_data = collections.defaultdict(
        lambda: collections.defaultdict(dict))

    for (obj, vertices) in profiled_vertices.iteritems():
        for v in vertices:
            for subvertex in v.subvertices:
                profile_data[obj][v][subvertex] =\
                    get_subvertex_profile_data(txrx, subvertex)

    return profile_data


def get_subvertex_profile_data(txrx, subvertex):
    """Get profiling data from the given subvertex."""
    raise NotImplementedError
