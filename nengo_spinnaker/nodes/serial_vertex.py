from pacman103.front import common


class SerialVertex(common.ExternalDeviceVertex):
    size_in = None
    def __init__(self,
                 virtual_chip_coords=dict(x=0xFE, y=0xFF),
                 connected_node_coords=dict(x=1, y=0),
                 connected_node_edge=common.edges.EAST):
        super(SerialVertex, self).__init__(
            n_neurons=0,
            virtual_chip_coords=virtual_chip_coords,
            connected_node_coords=connected_node_coords,
            connected_node_edge=connected_node_edge
        )

    def generate_routing_info(self, subedge):
        # TODO When PACMAN is refactored we can get rid of this because we've
        #      already allocated keys to connections, and there is a map of 1
        #      connection to 1 edge and keys are placement independent (hence
        #      all subedges of an edge share a key).
        return (subedge.edge.keyspace.routing_key(c=0),
                subedge.edge.keyspace.routing_mask)
