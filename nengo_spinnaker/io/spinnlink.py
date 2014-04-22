from . import serial_vertex
from .. import filter_vertex


class SpiNNlinkUSB(object):
    def __init__(self):
        self._serial_vertex = serial_vertex.SerialVertex()

    def build_node(self, builder, node):
        """Build the given Node
        """
        pass

    def get_node_in_vertex(self, builder, c):
        """Get the Vertex which accepts input for the Node
        """
        # Create a Filter vertex to relay data out to SpiNNlink
        postvertex = filter_vertex.FilterVertex(
            c.post.size_in, output_id=0, output_period=10)
        builder.add_vertex(postvertex)

        # Create an edge from the Filter vertex to the Serial Vertex
        edge = edges.NengoEdge(c, postvertex, self._serial_vertex)
        builder.add_edge(edge)

        # Return the Filter vertex
        return postvertex

    def get_node_out_vertex(self, builder, c):
        """Get the Vertex which transmits output from the Node
        """
        return self._serial_vertex

    def get_node_input(self, node):
        raise NotImplementedError

    def set_node_output(self, node):
        raise NotImplementedError
