from . import serial_vertex
from .. import filter_vertex, edges
from . import io_builder


class SpiNNlinkUSB(io_builder.IOBuilder):
    def __init__(self):
        self._serial_vertex = serial_vertex.SerialVertex()

    def build_node(self, builder, node):
        """Build the given Node
        """
        pass

    def get_node_in_vertex(self, c):
        """Get the Vertex for input to the terminating Node of the given
        Connection
        """
        # Create a Filter vertex to relay data out to SpiNNlink
        postvertex = filter_vertex.FilterVertex(
            c.post.size_in, output_id=0, output_period=10)
        self.builder.add_vertex(postvertex)

        # Create an edge from the Filter vertex to the Serial Vertex
        edge = edges.NengoEdge(c, postvertex, self._serial_vertex)
        self.builder.add_edge(edge)

        # Return the Filter vertex
        return postvertex

    def get_node_out_vertex(self, c):
        """Get the Vertex for output from the originating Node of the given
        Connection
        """
        return self._serial_vertex
