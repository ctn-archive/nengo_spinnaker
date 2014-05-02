import numpy as np
import socket
import struct
import threading
import time

from pacman103.lib import parameters
from pacman103.core.spinnman.sdp import sdp_message as sdp

from . import receive_vertex, transmit_vertex
from . import io_builder
from .. import utils


class Ethernet(io_builder.IOBuilder):
    """IO handler for Ethernet connectivity to a SpiNNaker board.

    Handles:
        * Building `Node`s (adding appropriate executables to read inputs and
          set outputs.
        * Getting and setting `Node` inputs and outputs.
    """
    def __init__(self, machinename, port=17895, input_period=10./32):
        self.machinename = machinename
        self.port = port
        self.input_period = input_period

        self._tx_vertices = list()
        self._tx_assigns = dict()
        self._rx_vertices = list()
        self._rx_assigns = dict()

        self.node_to_node_edges = list()
        self.nodes = list()

    def build_node(self, builder, node):
        """Build the given Node
        """
        self.nodes.append(node)

        # If the Node has input, then assign the Node to a Tx component
        if node.size_in > 0:
            tx = transmit_vertex.TransmitVertex(
                node=node,
                label="Tx for %s" % node
            )
            builder.add_vertex(tx)
            self._tx_assigns[node] = tx
            self._tx_vertices.insert(0, tx)

        # If the Node has output, and that output is not constant, then assign
        # the Node to an Rx component.
        if node.size_out > 0 and callable(node.output):
            # Try to fit the Node in an existing Rx Element
            # Most recently added Rxes are nearer the start
            for rx in self._rx_vertices:
                if rx.remaining_dimensions >= node.size_out:
                    rx.assign_node(node)
                    self._rx_assigns[node] = rx
                    break
            else:
                # Otherwise create a new Rx element
                rx = receive_vertex.ReceiveVertex(
                    label="Rx%d" % len(self._rx_vertices)
                )
                builder.add_vertex(rx)
                rx.assign_node(node)
                self._rx_assigns[node] = rx
                self._rx_vertices.insert(0, rx)

    def get_node_in_vertex(self, builder, c):
        """Get the Vertex for input to the terminating Node of the given
        Connection

        :raises KeyError: if the Node is not built/known
        """
        return self._tx_assigns[c.post]

    def get_node_out_vertex(self, builder, c):
        """Get the Vertex for output from the originating Node of the given
        Connection

        :raises KeyError: if the Node is not built/known
        """
        return self._rx_assigns[c.pre]

    def __enter__(self):
        """Create and return a Communicator to handle input/output over
        Ethernet.

        The Communicator supports the functions `get_input(node)` and
        `set_output(node, value)`, and will perform IO at appropriate
        intervals.
        """
        self._comms = EthernetCommunicator(self._tx_assigns, self._tx_vertices,
                                           self._rx_assigns, self._rx_vertices,
                                           self.machinename, self.port,
                                           tx_period=self.input_period)
        return self._comms

    def __exit__(self, exc_type, exc_val, traceback):
        """Stop the Communicator."""
        self._comms.stop()


class EthernetCommunicator(object):
    """Manage retrieving the input and the setting the output values for Nodes
    """
    def __init__(self, tx_assigns, tx_vertices, rx_assigns, rx_vertices,
                 machinename, rx_port=17895, rx_period=0.00001,
                 tx_period=0.05):
        """Create a new EthernetCommunicator to handle communication with
        the given set of nodes.

        :param tx_assigns: dictionary mapping `Node`s to `TransmitVertex`s
        :param tx_vertices: iterable of `TransmitVertex`s
        :param rx_assigns: dictionary mapping `Node`s to `ReceiveVertex`s
        :param rx_vertices: iterable of `ReceiveVertex`s
        :param machinename: hostname of the SpiNNaker machine
        :param rx_port: port number on which to listen for incoming packets
        :param rx_period: period with which to poll the socket for incoming
                          packets
        :param tx_period: period with which to transmit output from `Node`s
        """
        # Prepare the input/output caches
        self._vals = dict([(node, None) for node in tx_assigns.keys()])
        self._output_cache = dict([
            (rx, np.zeros(rx.n_assigned_dimensions)) for rx in rx_vertices
        ])
        self._output_fresh = dict([(rx, False) for rx in rx_vertices])

        # Generate a mapping of (x, y, p) to Node
        self._node_coords = dict(
            [(tx.subvertices[0].placement.processor.get_coordinates(),
              tx.node) for tx in tx_vertices]
        )

        # Parameters
        self.machinename = machinename
        self.rx_period = rx_period
        self.tx_period = tx_period

        # Save Tx and Rx assignments and vertices
        self.tx_assigns = tx_assigns  # INPUT to Nodes
        self.rx_assigns = rx_assigns  # OUTPUT from Nodes
        self.tx_vertices = tx_vertices
        self.rx_vertices = rx_vertices

        # Generate sockets for IO
        self._out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._out_sock.setblocking(0)
        self._in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._in_sock.bind(("", rx_port))
        self._in_sock.setblocking(0)

        # Locks
        self._out_lock = threading.Lock()
        self._in_lock = threading.Lock()

        # Create and start the timers
        self.tx_timer = threading.Timer(self.tx_period, self.tx_tick)
        self.rx_timer = threading.Timer(self.rx_period, self.rx_tick)
        self.tx_timer.start()
        self.rx_timer.start()

    def stop(self):
        """Stop the communicator.

        Cancel the Timers and close the sockets.
        """
        self.tx_timer.cancel()
        self.rx_timer.cancel()
        self._out_sock.close()
        self._in_sock.close()

    def get_node_input(self, node):
        """Return the latest input for the given Node

        :return: an array of data for the Node, or None if no data received
        :raises KeyError: if the Node is not a valid Node
        """
        with self._in_lock:
            return self._vals[node]

    def set_node_output(self, node, output):
        """Set the output of the given Node

        :raises KeyError: if the Node is not a valid Node
        """
        rx = self.rx_assigns[node]  # Get the Rx element
        i = rx.node_index(node)  # The offset of this Node in the Rx element

        with self._out_lock:
            if self._output_cache[rx][i:i+node.size_out] != output:
                self._output_cache[rx][i:i+node.size_out] = output
                self._output_fresh[rx] = True

    def rx_tick(self):
        """Internal "thread" used to receive inputs from SpiNNaker

        .. todo::
            Allow Tx components to represent input for multiple Nodes
        """
        try:
            data = self._in_sock.recv(512)
            msg = sdp.SDPMessage(data)

            node = self._node_coords[(msg.src_x, msg.src_y, msg.src_cpu)]

            # Convert the data
            data = msg.data[16:]
            assert(not len(data) % 4)
            vals = [struct.unpack("I", data[n*4:n*4 + 4])[0] for
                    n in range(len(data) / 4)]
            values = [(v - 0x100000000) * 2**-15 if v & 0x80000000 else
                      v * 2**-15 for v in vals]

            # Save the data
            assert(len(vals) == node.size_in)
            with self._in_lock:
                self._vals[node] = values
        except IOError:  # There was no data to receive
            pass

        self.rx_timer = threading.Timer(self.rx_period, self.rx_tick)
        self.rx_timer.start()

    def tx_tick(self):
        """Internal "thread" used to transmit values to SpiNNaker"""
        try:
            for rx_vertex in self.rx_vertices:
                if not self._output_fresh[rx_vertex]:
                    continue

                (x, y, p) = \
                    rx_vertex.subvertices[0].placement.processor\
                    .get_coordinates()

                with self._out_lock:
                    vals = self._output_cache[rx_vertex]
                    self._output_fresh[rx_vertex] = False

                data = struct.pack(
                    "H14x%di" % rx_vertex.n_assigned_dimensions, 1,
                    *parameters.s1615(vals)
                )

                packet = sdp.SDPMessage(dst_x=x, dst_y=y, dst_cpu=p, data=data)
                self._out_sock.sendto(str(packet), (self.machinename, 17893))
                time.sleep(0.005)

            self.tx_timer = threading.Timer(self.tx_period, self.tx_tick)
            self.tx_timer.start()
        except KeyboardInterrupt:
            pass
