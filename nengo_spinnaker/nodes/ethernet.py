import collections
import numpy as np
import socket
import struct
import threading
import time

from pacman103.core.spinnman.sdp import sdp_message as sdp

from . import sdp_receive_vertex, sdp_transmit_vertex
from .. import utils
from ..utils import fp


NodeRx = collections.namedtuple('NodeRx', ['rx', 'transform', 'start', 'stop'])


class Ethernet(object):
    """IO handler for Ethernet connectivity to a SpiNNaker board.

    Handles:
        * Building `Node`s (adding appropriate executables to read inputs and
          set outputs).
        * Getting and setting `Node` inputs and outputs.
    """
    def __init__(self, machinename, port=17895, input_period=10./32):
        self.machinename = machinename
        self.port = port
        self.input_period = input_period

        self._tx_vertices = list()  # List of Tx vertices
        self._tx_assigns = dict()   # Map of Nodes to Tx vertices
        self._rx_vertices = list()  # List of Rx vertices
        self._rx_assigns = dict()   # Map of assigned Nodes and Transforms
                                    # to Rx vertices

    def build_node(self, builder, node):
        """Build the given Node
        """
        pass

    def get_node_in_vertex(self, builder, c):
        """Get the Vertex for input to the terminating Node of the given
        Connection

        :raises KeyError: if the Node is not built/known
        """
        node = c.post
        assert(node.size_in > 0)

        # If the Node already has a Tx, then return it
        if node in self._tx_assigns:
            return self._tx_assigns[node]

        # Otherwise create one
        tx = sdp_transmit_vertex.SDPTransmitVertex(
            node=node,
            label="Tx for %s" % node
        )
        builder.add_vertex(tx)
        self._tx_assigns[node] = tx
        self._tx_vertices.insert(0, tx)
        return tx

    def get_node_out_vertex(self, builder, c):
        """Get the Vertex for output from the originating Node of the given
        Connection, given the transform applied by this Connection.
        """
        nte = sdp_receive_vertex.NodeTransformEntry(c.pre, c.transform,
                                                utils.get_connection_width(c))

        # See if the combination of this Node and transform has already been
        # assigned
        if nte in self._rx_assigns:
            return self._rx_assigns[nte]

        # Try and fit the Node/Transform combination in an existing Rx element,
        # otherwise create a new one
        for rx in self._rx_vertices:
            if nte.width <= rx.n_remaining_dimensions:
                self._rx_assigns[nte] = rx
                rx.add_node_transform(nte.node, nte.transform)
                return rx
        else:
            rx = sdp_receive_vertex.SDPReceiveVertex(
                label="Rx%d" % len(self._rx_vertices)
            )
            rx.add_node_transform(nte.node, nte.transform, nte.width)
            builder.add_vertex(rx)
            self._rx_assigns[nte] = rx
            self._rx_vertices.insert(0, rx)
            return rx

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
        :param rx_assigns: dictionary mapping `Node`s and Transforms to
                           `ReceiveVertex`s
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

        # Generate a mapping of Nodes to Rxs, transforms and indices
        self._node_out = collections.defaultdict(dict)
        for (nte, rx) in rx_assigns.items():
            offset = rx.get_node_transform_offset(nte.node, nte.transform)
            nrx = NodeRx(rx, np.array(nte.transform),
                         offset, offset + nte.width)
            self._node_out[nte.node][nte.transform] = nrx

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
        self.tx_timer.name = "EthernetTx"
        self.rx_timer = threading.Timer(self.rx_period, self.rx_tick)
        self.rx_timer.name = "EthernetRx"
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
        for nrx in self._node_out[node].values():
            # Transform the output and cache
            t_output = np.dot(nrx.transform, output)

            with self._out_lock:
                if self._output_cache[nrx.rx][nrx.start:nrx.stop] != t_output:
                    self._output_cache[nrx.rx][nrx.start:nrx.stop] = t_output
                    self._output_fresh[nrx.rx] = True

    def rx_tick(self):
        """Internal "thread" used to receive inputs from SpiNNaker

        .. todo::
            Allow Tx components to represent input for multiple Nodes
        """
        try:
            data = self._in_sock.recv(512)
            msg = sdp.SDPMessage(data)

            try:
                node = self._node_coords[(msg.src_x, msg.src_y, msg.src_cpu)]
            except KeyError:
                raise Exception("Received packet from core (%3d, %3d, %2d). "
                                "No Node is assigned to this core. "
                                "Board may require rebooting.")

            # Convert the data
            data = msg.data[16:]
            assert(not len(data) % 4)
            vals = [struct.unpack("I", data[n*4:n*4 + 4])[0] for
                    n in range(len(data) / 4)]
            values = fp.kbits(vals)

            # Save the data
            assert(len(vals) == node.size_in)
            with self._in_lock:
                self._vals[node] = values
        except IOError:  # There was no data to receive
            pass

        self.rx_timer = threading.Timer(self.rx_period, self.rx_tick)
        self.rx_timer.name = "EthernetRx"
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
                    "H14x%dI" % rx_vertex.n_assigned_dimensions, 1,
                    *fp.bitsk(vals)
                )

                packet = sdp.SDPMessage(dst_x=x, dst_y=y, dst_cpu=p, data=data)
                self._out_sock.sendto(str(packet), (self.machinename, 17893))
                time.sleep(0.005)

            self.tx_timer = threading.Timer(self.tx_period, self.tx_tick)
            self.tx_timer.name = "EthernetTx"
            self.tx_timer.start()
        except KeyboardInterrupt:
            pass
