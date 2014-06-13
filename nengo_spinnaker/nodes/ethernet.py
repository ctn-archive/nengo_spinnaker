import collections
import logging
import numpy as np
import socket
import struct
import threading

from pacman103.core.spinnman.sdp import sdp_message as sdp

from nengo_spinnaker.nodes.sdp_receive_vertex import SDPReceiveVertex
from nengo_spinnaker.nodes.sdp_transmit_vertex import SDPTransmitVertex
from nengo_spinnaker.utils import fp

logger = logging.getLogger(__name__)

ConnectionRx = collections.namedtuple('ConnectionRxes', ['rx', 'connection'])
BufferedConnection = collections.namedtuple(
    'BufferedConnection', ['rx', 'connection', 'transform', 'buffered_output'])


def stop_on_keyboard_interrupt(f):
    def f_(self, *args):
        try:
            f(self, *args)
        except KeyboardInterrupt:
            self.stop()
    return f_


class Ethernet(object):
    """Ethernet communicator and Node builder."""

    def __init__(self, machinename, port=17895, input_period=10./32):
        # General parameters
        self.machinename = machinename
        self.port = port
        self.input_period = input_period
        self.comms = None

        # Map Node --> Tx
        self.nodes_txes = dict()

        # Map Node --> unique Conns
        self.nodes_connections = collections.defaultdict(list)

        # Map Node --> [Rx]
        self.nodes_rxes = collections.defaultdict(list)

        # Map unique Connections --> Rx
        self.connections_rx = dict()

    def build_node(self, builder, node):
        """Do nothing, we build Node IO when connecting to/from a Node."""
        pass

    def get_node_in_vertex(self, builder, conn):
        """Return the Vertex at which to terminate the given x->Node connection

        Nodes receive inputs from Tx vertices, see if a Tx vertex has already
        been assigned for this Node: if so, then return it, otherwise create
        a new Tx vertex and return that.

        .. todo::
            Modify Tx vertices so that they can receive input for multiple
            Nodes.
        """
        # Have we already assigned a Tx vertex for this Node
        if conn.post in self.nodes_txes:
            return self.node_txes[conn.post]

        # Create a new Tx, add to the map, add to the graph
        tx = SDPTransmitVertex(conn.post, label="SDP_TX %s" % conn.post)
        self.nodes_txes[conn.post] = tx
        builder.add_vertex(tx)
        return tx

    def get_node_out_vertex(self, builder, conn):
        """Return the Vertex at which to originate the given Node->x connection

        Nodes transmit output to Rx vertices.  See if an Rx vertex with a
        compatible connection has already been assigned.  If so, return it,
        otherwise create a new Rx vertex and return that.  Also generate a
        list of unique connections (i.e., those not found compatible).
        """
        rx = None
        for _rx in self.nodes_rxes[conn.pre]:
            # See if we've already assigned a similar connection for this Node
            if rx.contains_compatible_connection(conn):
                rx = _rx
                break
        else:
            # Remember that this is a unique connection
            self.nodes_connections[conn.pre].append(conn)

            # See if we have space for this connection in any of the existing
            # Rxes for this Node
            for _rx in self.nodes_rxes[conn.pre]:
                if _rx.n_remaining_dimensions >= conn.size_out:
                    rx = _rx
                    break
            else:
                rx = SDPReceiveVertex()
                self.nodes_rxes[conn.pre].insert(0, rx)
                builder.add_vertex(rx)

            # Record the association between this unique connection and the Rx
            self.connections_rx[conn] = rx

        # Add the connection to the selected Rx, return it
        rx.add_connection(conn)
        return rx

    @property
    def io(self):
        if self.comms is None:
            self.comms = EthernetCommunicator()
        return self.comms

    def __enter__(self):
        # Generates a map of Nodes --> [(Rx, Connection)]
        nodes_connections_rxs = collections.defaultdict(list)
        for (node, connections) in self.nodes_connections.items():
            for c in connections:
                nodes_connections_rxs[node].append(
                    ConnectionRx(rx=self.connections_rx[c], connection=c)
                )

        self.comms.setup(self.machinename, self.port, self.input_period,
                         self.nodes_txes, nodes_connections_rxs)
        return self.comms

    def __exit__(self, exc_type, exc_val, traceback):
        self.comms.stop()


class EthernetCommunicator(object):
    def __init__(self):
        """Create the object so that we have reference to it, setup has to be
        later.
        """
        pass

    def setup(self, machinename, port, input_period, nodes_tx,
              nodes_connections_rxs):
        """Create a new EthernetCommunicator."""
        self.machinename = machinename
        self.port = port
        self.input_period = input_period
        self.nodes_tx = nodes_tx
        self.nodes_connection_rxs = nodes_connections_rxs

        # Generate a list of Rxs, then generate an output buffer for each,
        # finally slice that buffer and store references with Nodes and
        # Connections
        self.rxs = list()
        self.rx_fresh = dict()
        self.rx_buffer = dict()
        self.nodes_connections_buffers = collections.defaultdict(list)
        self._node_out = list()  # List of Nodes which provide output

        for (node, connections_rxs) in nodes_connections_rxs.items():
            if node not in self._node_out:
                self._node_out.append(node)

            for crx in connections_rxs:
                # If we haven't seen this Rx element before, then add it to the
                # set of Rx elements, create a fresh mark and a buffer
                if crx.rx not in self.rxs:
                    self.rxs.append(crx.rx)
                    self.rx_fresh[crx.rx] = False
                    self.rx_buffer[crx.rx] = np.zeros(
                        crx.rx.n_assigned_dimensions)

                # Offset into the Rx buffer and slice
                offset = crx.rx.get_connection_offset(crx.connection)
                cbuffer = self.rx_buffer[crx.rx][
                    offset:offset + crx.connection.post.size_in]

                # Store the Rx, Connection and Buffer slice
                i = crx.rx.connections[crx.connection]
                _transform = crx.rx.connections.transforms_functions[i].transform
                self.nodes_connections_buffers[node].append(
                    BufferedConnection(crx.rx, crx.connection, _transform,
                                       cbuffer)
                )

        # Generate a map of x, y, p to Node for received input, a cache of Node
        # input
        self.xyp_nodes = dict()
        self.node_inputs = dict()
        for (node, tx) in self.nodes_tx.items():
            xyp = tx.subvertices[0].placement.processor.get_coordinates()
            self.xyp_nodes[xyp] = node
            self.node_inputs[node] = None

        # Sockets
        self.in_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.in_socket.setblocking(0)
        self.in_socket.bind(("", self.port))

        self.out_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.out_socket.setblocking(0)

        # Locks
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()

        # Tx, Rx timers
        self.tx_period = self.input_period
        self.rx_period = 0.0005
        self.rx_timer = threading.Timer(self.rx_period, self.sdp_rx_tick)
        self.rx_timer.name = "EthernetRx"
        self.tx_timer = threading.Timer(self.tx_period, self.sdp_tx_tick)
        self.tx_timer.name = "EthernetTx"

    def start(self):
        self.tx_timer.start()
        self.rx_timer.start()

    def stop(self):
        self.tx_timer.cancel()
        self.rx_timer.cancel()
        self.in_socket.close()
        self.out_socket.close()

    def node_has_output(self, node):
        """Return whether the given Node has output"""
        return node in self._node_out

    def get_node_input(self, node):
        """Get the input for the given Node.

        :return: Latest input for the given Node or None if not input has been
                 received.
        :raises: :py:exc:`KeyError` if the Node is not recognised.
        """
        with self.input_lock:
            return self.node_inputs[node]

    def set_node_output(self, node, output):
        """Set the output for the given Node.

        :raises: :py:exc:`KeyError` if the Node is not recognised.
        """
        # For each unique connection
        for crxb in self.nodes_connections_buffers[node]:
            # Transform the output, store in the buffer and mark the Rx as
            # being fresh.
            if callable(crxb.connection.function):
                output = crxb.connection.function(output)
            output = np.dot(crxb.transform, output)

            if np.any(output != crxb.buffered_output):
                with self.output_lock:
                    crxb.buffered_output[:] = output.reshape(output.size)
                    self.rx_fresh[crxb.rx] = True

    @stop_on_keyboard_interrupt
    def sdp_tx_tick(self):
        """Transmit packets to the SpiNNaker board.
        """
        # Look for Rx elements with fresh output, transmit the output and
        # mark as stale.
        for rx in self.rxs:
            if self.rx_fresh[rx]:
                xyp = rx.subvertices[0].placement.processor.get_coordinates()

                with self.output_lock:
                    data = fp.bitsk(self.rx_buffer[rx])
                    self.rx_fresh[rx] = False

                data = struct.pack("H14x%dI" % rx.n_assigned_dimensions, 1,
                                   *data)
                packet = sdp.SDPMessage(dst_x=xyp[0], dst_y=xyp[1],
                                        dst_cpu=xyp[2], data=data)
                self.out_socket.sendto(str(packet), (self.machinename, 17893))

        # Reschedule the Tx tick
        self.tx_timer = threading.Timer(self.tx_period, self.sdp_tx_tick)
        self.tx_timer.name = "EthernetTx"
        self.tx_timer.start()

    @stop_on_keyboard_interrupt
    def sdp_rx_tick(self):
        """Receive packets from the SpiNNaker board.
        """
        try:
            data = self.in_socket.recv(512)
            msg = sdp.SDPMessage(data)

            try:
                node = self.xyp_nodes[(msg.src_x, msg.src_y, msg.src_cpu)]
            except KeyError:
                logger.error(
                    "Received packet from unexpected core (%3d, %3d, %3d). "
                    "Board may require resetting." %
                    (msg.src_x, msg.src_y, msg.src_cpu)
                )
                raise IOError  # Jumps out of the receive logic

            # Convert the data
            data = msg.data[16:]
            vals = [struct.unpack("I", data[n*4:n*4 + 4])[0] for n in
                    range(len(data)/4)]
            values = fp.kbits(vals)

            # Save the data
            assert(len(values) == node.size_in)
            with self.input_lock:
                self.node_inputs[node] = values
        except IOError:
            pass

        # Reschedule the Rx tick
        self.rx_timer = threading.Timer(self.rx_period, self.sdp_rx_tick)
        self.rx_timer.name = "EthernetRx"
        self.rx_timer.start()
