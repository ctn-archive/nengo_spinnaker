import serial
import threading

from . import serial_vertex
from .. import filter_vertex, edges
from ..utils import fp


class SpiNNlinkUSB(object):
    def __init__(self, device):
        self._serial_vertex = None
        self.device = device

    def build_node(self, builder, node):
        """Build the given Node
        """
        pass

    def get_serial_vertex(self, builder):
        """Get (or create) the serial vertex."""
        if self._serial_vertex is None:
            self._serial_vertex = serial_vertex.SerialVertex()
            builder.add_vertex(self._serial_vertex)
        return self._serial_vertex

    def get_node_in_vertex(self, builder, c):
        """Get the Vertex for input to the terminating Node of the given
        Connection
        """
        # Create a Filter vertex to relay data out to SpiNNlink
        postvertex = filter_vertex.FilterVertex(
            c.post.size_in, output_id=0, output_period=10)
        builder.add_vertex(postvertex)

        # Create an edge from the Filter vertex to the Serial Vertex
        serial_vertex = self.get_serial_vertex(builder)
        edge = edges.NengoEdge(c, postvertex, serial_vertex)
        builder.add_edge(edge)

        # Return the Filter vertex
        return postvertex

    def get_node_out_vertex(self, builder, c):
        """Get the Vertex for output from the originating Node of the given
        Connection
        """
        return self.get_serial_vertex(builder)

    def generate_serial_key_maps(self):
        """Generate a map from incoming keys to Nodes, and from outgoing Nodes
        to keys.
        """
        serial_rx = {}
        serial_tx = {}

        for edge in self._serial_vertex.in_edges:
            for subedge in edge.subedges:
                key = edge.prevertex.generate_routing_info(subedge)[0]
                node = edge.post
                serial_tx[key] = node

        for edge in self._serial_vertex.out_edges:
            for subedge in edge.subedges:
                key = edge.prevertex.generate_routing_info(subedge)[0]
                node = edge.pre
                if node not in serial_rx:
                    serial_rx[node] = [key]
                else:
                    serial_rx[node].append(key)

        return serial_tx, serial_rx

    def __enter__(self):
        self.communicator = SpiNNlinkUSBCommunicator(
            *self.generate_serial_key_maps(), dev=self.device)
        return self.communicator

    def __exit__(self, exc_type, exc_val, trace):
        self.communicator.stop()


class SpiNNlinkUSBCommunicator(object):
    """Manage retrieving the input and the setting the output values for Nodes
    """
    def __init__(self, serial_tx, serial_rx, dev, rx_period=0.0001):
        """Create a new SpiNNlinkUSBCommunicator to handle communication with
        the given set of nodes and key mappings.

        :param serial_tx: map of keys to `Node`s - input
        :param serial_rx: map of `Node`s to keys - output
        :param dev: serial port to use
        :param rx_period: period between polling for input
        """
        self._vals = dict([(node, None) for node in serial_tx.values()])
        self.serial_rx = serial_rx
        self.serial_tx = serial_tx
        self.rx_period = rx_period

        # Create the Serial connection
        self.serial = serial.Serial(dev, baudrate=8000000, rtscts=True,
                                    timeout=rx_period)
        self.serial.write("S+\n")  # Send SpiNNaker packets to host

        # Start the thread(s)
        self.rx_timer = threading.Timer(self.rx_period, self.rx_tick)
        self.rx_timer.name = "SpiNNlinkUSBRx"

    def start(self):
        self.rx_timer.start()

    def stop(self):
        self.rx_timer.cancel()

    def node_has_output(self, node):
        """Return whether the given Node has output"""
        return node in self.serial_rx

    def get_node_input(self, node):
        """Return the latest input for the given Node

        :return: an array of data for the Node, or None if no data received
        :raises KeyError: if the Node is not a valid Node
        """
        return self._vals[node]

    def set_node_output(self, node, output):
        """Set the output of the given Node

        Transmits the given value to the simulation.
        :raises KeyError: if the Node is not a valid Node
        """
        for key in self.serial_rx[node]:
            for d, v in enumerate(output):
                value = fp.bitsk(v)

                msg = "%08x.%08x\n" % (key | d, value)
                self.serial.write(msg)

        self.serial.flush()

    def rx_tick(self):
        """Internal "thread" for reading from serial connection."""
        try:
            data = self.serial.readline()

            if '.' in data:
                parts = [int(p, 16) for p in data.split('.')]
                if len(parts) == 3:
                    (header, key, payload) = parts

                value = fp.kbits(payload)

                base_key = key & 0xFFFFF800
                d = (key & 0x0000003F)

                if base_key in self.serial_tx:
                    node = self.serial_tx[base_key]

                    if self._vals[node] is None:
                        self._vals = [None] * node.size_in

                    self._vals[self.serial_tx[base_key]][d] = value
        except IOError:  # No data to read
            print "IOError"

        self.rx_timer = threading.Timer(self.rx_period, self.rx_tick)
        self.rx_timer.name = "SpiNNlinkUSBRx"
        self.rx_timer.start()
