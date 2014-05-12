from pacman103.front import common


class SerialVertex(common.ExternalDeviceVertex):
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
        # We use the virtual chip co-ordinates to fill in part of the key, the
        # ID is taken as the index of this edge from the serial vertex.
        # As a result, there is a limitation to the number of things which may
        # be fed from a serial vertex (above and beyond the obvious bandwidth
        # limitations).
        x = self.virtual_chip_coords['x']
        y = self.virtual_chip_coords['y']
        i = self.out_edges.index(subedge.edge)

        key = (x << 24) | (y << 16) | (i << 6)
        return key, 0xFFFFFFE0
