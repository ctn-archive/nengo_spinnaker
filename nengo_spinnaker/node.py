import numpy as np

import nengo

import utils


def replace_function_of_time_nodes(objs, conns, config, time_in_seconds, dt):
    """Replace function-of-time Nodes with the appropriate structure for
    simulation.
    """
    new_objs = list()
    new_conns = list()

    replaced_nodes = dict()
    for obj in objs:
        if isinstance(obj, nengo.Node):
            if config[obj].f_of_t:
                # Get the likely size of this object
                out_conns = utils.connections.Connections(c for c in conns if
                                                          c.pre_obj is obj)
                width = out_conns.width

                # Get the overall duration of the signal
                p_durations = [t for t in [time_in_seconds,
                                           config[obj].f_period] if
                               t is not None]

                if len(p_durations) == 0:
                    # Indefinite simulation with indefinite function, will
                    # have to simulate on host.
                    config[obj].f_of_t = False
                    new_objs.append(obj)
                    continue

                duration = min(p_durations)
                periodic = (config[obj].f_period is not None and
                            config[obj].f_period == duration)

                if width * duration > 6 * 1024**2:
                    # Storing this function (and all its transforms) would
                    # take up too much memory, will have to simulate on
                    # host.
                    # TODO Split up the connections to reduce the memory
                    #      usage instead of giving up.
                    config[obj].f_of_t = False
                    new_objs.append(obj)
                    continue

                # It is possible to fit the function (and all its
                # transforms) in memory, so replace it with a
                # function of time vertex.
                new_obj = ValueSource.from_node(
                    obj.output, out_conns, duration, periodic, dt)
                replaced_nodes[obj] = new_obj
                new_objs.append(new_obj)
            else:
                new_objs.append(obj)
        else:
            new_objs.append(obj)

    for c in conns:
        if c.pre_obj in replaced_nodes:
            c.pre_obj = replaced_nodes[c.pre_obj]
        new_conns.append(c)

    return new_objs, new_conns


class IntermediateFilter(object):
    def __init__(self, size_in, transmission_period=10):
        self.size_in = size_in
        self.transmission_period = transmission_period


class FilterVertex(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_filter'
    MAX_ATOMS = 1

    def __init__(self, size_in, in_connections, dt, output_period=100,
                 interpacket_pause=1):
        super(FilterVertex, self).__init__(1)
        self.size_in = size_in

        # Create the system region
        system_region = utils.vertices.UnpartitionedListRegion([
            size_in, None, 1000, output_period, interpacket_pause])

        # Create the filter regions
        (in_filters, in_routing) = utils.vertices.make_filter_regions(
            in_connections, dt)
        self.regions = [system_region, None, in_filters, in_routing, None]

    @classmethod
    def get_output_keys_region(cls, fv, assembler):
        output_keys = list()

        for c in assembler.get_outgoing_connections(fv):
            for d in range(c.width):
                output_keys.append(c.keyspace.key(d=d))

        return utils.vertices.UnpartitionedListRegion(output_keys)

    @classmethod
    def get_transform(cls, fv, assembler):
        # Combine the outgoing connections
        conns = utils.connections.Connections(
            assembler.get_outgoing_connections(fv))

        for tf in conns.transforms_functions:
            assert tf.function is None

        transforms = np.vstack(t.transform for t in conns.transforms_functions)
        transform_region = utils.vertices.UnpartitionedMatrixRegion(
            transforms, formatter=utils.fp.bitsk)

        return transforms.shape[0], transform_region

    @classmethod
    def assemble(cls, fv, assembler):
        if len(assembler.get_outgoing_connections(fv)) == 0:
            return None
        # Create the output keys region and add it to the instance, then
        # return.
        fv.regions[0].data[1], fv.regions[4] = cls.get_transform(fv, assembler)
        fv.regions[1] = cls.get_output_keys_region(fv, assembler)
        return fv

    @classmethod
    def assemble_from_intermediate(cls, fv, assembler):
        if len(assembler.get_outgoing_connections(fv)) == 0:
            return None
        # Create the vertex, then assemble that and return
        in_conns = utils.connections.Connections(
            assembler.get_incoming_connections(fv))

        fv_ = cls(fv.size_in, in_conns, assembler.dt,
                  output_period=fv.transmission_period)
        fv_.regions[1] = cls.get_output_keys_region(fv, assembler)
        fv_.regions[0].data[1], fv_.regions[4] =\
            cls.get_transform(fv, assembler)

        return fv_


class ValueSource(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_value_source'
    MAX_ATOMS = 1

    def __init__(self, system_region, keys_region, data_region):
        super(ValueSource, self).__init__(1)
        self.regions = [system_region, keys_region, data_region]

    @classmethod
    def from_node(cls, fn, conns, duration, periodic, dt):
        # Generate some evaluation points, construct the signal for the given
        # duration.
        ts = np.arange(0, duration, dt)
        data = []
        for t in ts:
            v = np.array(fn(t))

            output = []
            for tf in conns.transforms_functions:
                output.append(np.asarray(
                    np.dot(tf.transform, v if tf.function is None
                           else tf.function(v))))

            data.append(np.vstack(output))
        data = np.array(data)
        data.shape = (1, data.size)

        # Calculate the number of blocks
        frames_per_block = 5*1024/conns.width
        full_blocks = int(duration / dt) / frames_per_block
        partial_block = int(duration / dt) % frames_per_block

        # Prepare the system region, etc.
        system_items = [
            1000,
            conns.width,
            0x1 if periodic else 0x0,
            full_blocks,
            frames_per_block,
            partial_block
        ]
        system_region = utils.vertices.UnpartitionedListRegion(system_items)

        output_keys = utils.connections.get_output_keys(conns)
        output_keys_region = utils.vertices.UnpartitionedListRegion(
            output_keys)

        data_region = utils.vertices.MatrixRegionPartitionedByRows(
            data, in_dtcm=False, formatter=utils.fp.bitsk)

        return cls(system_region, output_keys_region, data_region)
