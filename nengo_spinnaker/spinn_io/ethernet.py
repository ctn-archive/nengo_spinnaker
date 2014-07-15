import collections
import logging
import numpy as np
import socket
import struct
import threading

import nengo

from pacman103.core.spinnman.sdp import sdp_message as sdp

from nengo_spinnaker.utils import fp
from nengo_spinnaker import assembler, utils

logger = logging.getLogger(__name__)


def stop_on_keyboard_interrupt(f):
    def f_(self, *args):
        try:
            f(self, *args)
        except KeyboardInterrupt:
            self.stop()
    return f_


class TransformFunctionCollection(object):
    def __init__(self, outkeys):
        self.outkeys = outkeys
        self._tfs = list()

    def append(self, transform_function):
        # Generate the output keys for the transform/function
        for d in range(transform_function.transform.shape[0]):
            self.outkeys.append(transform_function.keyspace.key(d=d))

        # Store and reduce the remaining space
        self._tfs.append(transform_function)

    def __getitem__(self, i):
        return self._tfs[i]


class SDPRxVertex(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_rx'
    MAX_ATOMS = 1

    def __init__(self):
        super(SDPRxVertex, self).__init__(1)
        self.output_keys = list()
        self.transforms_functions = TransformFunctionCollection(
            self.output_keys)
        self.regions = list()

    @property
    def remaining_dims(self):
        return 64 - sum(
            [c.transform.shape[0] for c in self.transforms_functions])

    @classmethod
    def assemble(cls, rx, assembler):
        # Create the regions and monkey-patch them into the SDPRxVertex
        system_items = [1000, 64-rx.remaining_dims]
        system_region = utils.vertices.UnpartitionedListRegion(system_items)
        output_keys_region =\
            utils.vertices.UnpartitionedListRegion(rx.output_keys)

        rx.regions.extend([system_region, output_keys_region])

        return rx

assembler.Assembler.register_object_builder(SDPRxVertex.assemble, SDPRxVertex)


class SDPTxVertex(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_tx'
    MAX_ATOMS = 1

    def __init__(self, size_in, in_connections, dt, output_period=100):
        super(SDPTxVertex, self).__init__(1)
        """Create a new SDPTxVertex.

        :param size_in: The number of dimensions to accept.
        :param in_connections: A list of connections arriving at the Tx vertex.
        :param dt: Time step of the simulation.
        :param output_period: Period with which to transmit SDP packets (in
                              ticks)
        """
        # Construct the data to be loaded onto the board
        system_items = [size_in, 1000, output_period]
        system_region = utils.vertices.UnpartitionedListRegion(system_items)
        (input_filters, input_filter_routing) =\
            utils.vertices.make_filter_regions(in_connections, dt)

        # Create the regions
        self.regions = [system_region, input_filters, input_filter_routing]


class Ethernet(object):
    """Ethernet communicator and Node builder."""

    def __init__(self, machinename, port=17895, input_period=10./32):
        # General parameters
        self.machinename = machinename
        self.port = port
        self.input_period = input_period
        self.comms = None

        self.rx_elements = list()

        # Map Node --> Tx
        self.nodes_tx = dict()

        # Map Node --> transform, function, buffer index, rx
        self.nodes_connections = collections.defaultdict(list)

        # Map Rx --> Fresh
        self.rx_fresh = dict()
        self.rx_buffers = collections.defaultdict(list)

    @property
    def io(self):
        return self

    def prepare_network(self, objects, connections, dt, keyspace):
        """Swap out each Node with appropriate IO objects."""
        new_objs = list()
        new_conns = list()

        for obj in objects:
            # For each Node, combine outgoing connections
            if not isinstance(obj, nengo.Node):
                # If not a Node then retain the object
                new_objs.append(obj)
                continue

            out_conns = [c for c in connections if c.pre_obj == obj and
                         not isinstance(c.post_obj, nengo.Node)]
            outgoing_conns = utils.connections.Connections(out_conns)

            # Assign each unique combination of transform/function/keyspace to
            # a SDPRxVertex.
            for i, tfk in enumerate(outgoing_conns.transforms_functions):
                assert tfk.keyspace.is_set_i
                for rx in self.rx_elements:
                    if rx.remaining_dims >= tfk.transform.shape[0]:
                        break
                else:
                    rx = SDPRxVertex()
                    self.rx_elements.append(rx)
                    self.rx_fresh[rx] = False
                    new_objs.append(rx)

                rx.transforms_functions.append(tfk)
                buf = np.zeros(tfk.transform.shape[0])
                self.nodes_connections[obj].append((tfk, buf, rx))
                self.rx_buffers[rx].append(buf)

                # Replace the pre_obj on all connections from this Node to account
                # for the change to the SDPRxVertex.
                for c in out_conns:
                    if outgoing_conns[c] == i:
                        c.pre_obj = rx
                        c.is_accumulatory = False
                        new_conns.append(c)

            # Provide a Tx element to receive input for the Node
            in_conns = [c for c in connections if c.post_obj == obj and
                        not isinstance(c.pre_obj, nengo.Node)]
            if len(in_conns) > 0:
                tx = SDPTxVertex(obj.size_in, in_conns, dt)
                self.nodes_tx[obj] = tx
                new_objs.append(tx)

                for c in in_conns:
                    c.post_obj = tx
                    new_conns.append(c)

        # Retain all other connections unchanged
        for c in connections:
            if not (isinstance(c.pre_obj, nengo.Node) or
                    isinstance(c.post_obj, nengo.Node)):
                new_conns.append(c)

        return new_objs, new_conns

    def __enter__(self):
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
        self.stop_now = False
        self.tx_period = self.input_period
        self.rx_period = 0.0005
        self.rx_timer = threading.Timer(self.rx_period, self.sdp_rx_tick)
        self.rx_timer.name = "EthernetRx"
        self.tx_timer = threading.Timer(self.tx_period, self.sdp_tx_tick)
        self.tx_timer.name = "EthernetTx"

        return self

    def start(self):
        self.tx_timer.start()
        self.rx_timer.start()

    def stop(self):
        self.stop_now = True

        self.tx_timer.cancel()
        self.rx_timer.cancel()
        self.in_socket.close()
        self.out_socket.close()

    def __exit__(self, exc_type, exc_val, traceback):
        self.stop()

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
        # For each unique connection compute the output and store in the buffer
        for (tf, buf, rx) in self.nodes_connections[node]:
            c_output = output
            if tf.function is not None:
                c_output = tf.function(c_output)
            buf[:] = np.dot(tf.transform, c_output)
            self.rx_fresh[rx] = True

    @stop_on_keyboard_interrupt
    def sdp_tx_tick(self):
        """Transmit packets to the SpiNNaker board.
        """
        # Look for Rx elements with fresh output, transmit the output and
        # mark as stale.
        for rx in self.rx_elements:
            if self.rx_fresh[rx]:
                xyp = rx.subvertices[0].placement.processor.get_coordinates()

                with self.output_lock:
                    data = fp.bitsk(np.hstack(self.rx_buffers[rx]))
                    self.rx_fresh[rx] = False

                data = struct.pack("H14x%dI" % len(data), 1, *data)
                packet = sdp.SDPMessage(dst_x=xyp[0], dst_y=xyp[1],
                                        dst_cpu=xyp[2], data=data)
                self.out_socket.sendto(str(packet), (self.machinename, 17893))

        # Reschedule the Tx tick
        if not self.stop_now:
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
        if not self.stop_now:
            self.rx_timer = threading.Timer(self.rx_period, self.sdp_rx_tick)
            self.rx_timer.name = "EthernetRx"
            self.rx_timer.start()
