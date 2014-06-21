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
