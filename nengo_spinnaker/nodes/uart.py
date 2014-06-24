"""Builders and communicators necessary for serial/USB communication.
"""

import collections
import numpy as np
import serial
import threading
import struct

from pacman103.front import common

from nengo_spinnaker.utils import fp
from .ethernet import stop_on_keyboard_interrupt
import serial_vertex
from .. import edges, filter_vertex
from ..utils import connections


class UART(object):
    """A builder and communicator for generic serial interfaces.

    :param protocol:
    :param connection: The location where the connection to SpiNNlink is made.
    :param **kwargs: Arguments for the protocol object
    """
    def __init__(self, protocol, connection=None,
                 virtual_chip_coords={"x": 0xFF, "y": 0xFE},
                 connected_node_coords={"x": 0, "y": 0},
                 connected_node_edge=common.edges.SOUTH_WEST,
                 **kwargs):
        self.virtual_chip_coords = virtual_chip_coords
        self.connected_node_coords = connected_node_coords
        self.connected_node_edge = connected_node_edge

        # General components
        self.protocol = protocol(**kwargs)  # Should we instantiate this? TODO
        # self.connection = connection  # TODO
        self._serial_vertex = None
        self.base_key = ((self.virtual_chip_coords["x"] << 24)
                         | (self.virtual_chip_coords["y"] << 16))

        self.nodes_filters = dict()  # Map of Nodes to FilterVertices
        self.nodes_inputs = dict()  # Map of Nodes to input

    def _get_serial_vertex(self, builder):
        """Get (or create) the serial vertex."""
        if self._serial_vertex is None:
            self._serial_vertex = serial_vertex.SerialVertex(
                virtual_chip_coords=self.virtual_chip_coords,
                connected_node_coords=self.connected_node_coords,
                connected_node_edge=self.connected_node_edge,
                )
            builder.add_vertex(self._serial_vertex)
        return self._serial_vertex

    @property
    def io(self):
        return self

    def build_node(self, builder, node):
        # TODO: Probably deprecate and remove this function as it's not used...
        pass

    def get_node_in_vertex(self, builder, c):
        """Get the vertex which will accept input on this Connection.
        """
        # Create a new FilterVertex for the Node, unless one already exists in
        # which case return that
        node = c.post

        if c.post not in self.nodes_filters:
            # Create the new FilterVertex
            self.nodes_filters[node] = filter_vertex.FilterVertex(
                c.post.size_in, 0, output_period=10, interpacket_pause=50)
            builder.add_vertex(self.nodes_filters[node])

            # Connect it to the SerialVertex
            edge = edges.NengoEdge(
                c, self.nodes_filters[node], self._get_serial_vertex(builder))
            builder.add_edge(edge)

        # Return a reference to the FilterVertex
        return self.nodes_filters[node]

    def get_node_out_vertex(self, builder, c):
        """Get the vertex which will transmit output on this Connection.
        """
        # Return a reference to the serial vertex
        return self._get_serial_vertex(builder)

    def __enter__(self):
        # Prepare maps of Nodes->Connections
        # TODO: Neaten up the connection bank to make this easier!
        self.outgoing_connections = connections.ConnectionBank(
            [c for c in self._serial_vertex.out_edges])
        self.outgoing_ids = dict(
            [(c, i) for (i, c) in enumerate(self.outgoing_connections)])

        # Map of Keys->Nodes
        # Prepare buffers for Node inputs
        self.keys_nodes = dict()
        for (node, fv) in self.nodes_filters.items():
            # TODO: Neaten this up with KeySpaces!
            k = fv.generate_routing_info(
                fv.subvertices[0].out_subedges[0])[0]
            self.keys_nodes[k] = node

            self.nodes_inputs[node] = [None]*node.size_in

        return self

    def __exit__(self, *args):
        # Ensure everything shuts down ok
        self.protocol.stop()

    def start(self):
        # Start!
        self.protocol.start(self)

    def get_node_input(self, node):
        """Get the input for the Node or None if no (or incomplete) input has
        been received
        """
        if None in self.nodes_inputs[node]:
            return None
        return np.asarray(self.nodes_inputs[node])

    def set_node_output(self, node, output):
        """Set the output for the Node
        """
        # For each outgoing connection for the Node perform the appropriate
        # functions and transforms, then transmit packets for each dimension in
        # the output.
        # TODO: Neaten up the connection bank to make this easier!
        for c in self.outgoing_connections._connections[node]:
            t_output = output
            if c.function is not None:
                t_output = c.function(t_output)
            t_output = np.dot(c.transform, t_output)

            # Transmit the packets
            key = self.base_key + (self.outgoing_ids[c] << 6)
            for (d, v) in enumerate(t_output):
                self.protocol.queue_mc_packet(key | d, fp.bitsk(v))

    def receive_mc_packet(self, key, payload):
        """Handle an incoming MC packet, store the received dimension value."""
        # TODO: Deal with this properly rather than with handcoded masks
        node = self.keys_nodes[key & 0xffffffc0]
        d = key & 0x3f
        self.nodes_inputs[node][d] = fp.kbits(payload)


class GenericUARTProtocol(object):
    """GenericUARTProtocol provides the interface necessary to receive and
    transmit SpiNNaker packets over USB or UART connections.
    """
    def __init__(self):
        """Create (but do not start) a new GenericUARTProtocol handler."""
        self.outgoing_packet_queue = collections.OrderedDict()
        self.queue_lock = threading.Lock()

        self.stop_now = False

        self.transmit_ticker = threading.Timer(
            self.tx_period, self.transmit_tick)
        self.receive_ticker = threading.Timer(
            self.rx_period, self.receive_tick)

    def start(self, io):
        """Start the communication threads."""
        self.io = io  # Save a reference to the IO handler
        self.transmit_ticker.start()
        self.receive_ticker.start()

    def stop(self):
        """Stop the communication threads."""
        self.stop_now = True
        self.transmit_ticker.cancel()
        self.receive_ticker.cancel()

    def queue_mc_packet(self, key, payload):
        """Register a multicast packet in the queue."""
        with self.queue_lock:
            self.outgoing_packet_queue[key] = payload

    @stop_on_keyboard_interrupt
    def transmit_tick(self):
        """Transmit a single packet from the transmit queue and reschedule."""
        # If there are any packets to send then transmit a single packet
        key, payload = None, None
        with self.queue_lock:
            if len(self.outgoing_packet_queue) > 0:
                key, payload = self.outgoing_packet_queue.popitem(last=False)
        if key is not None:
            self.send_mc_packet(key, payload)

        # Schedule this function to run again
        if not self.stop_now:
            self.transmit_ticker = threading.Timer(
                self.tx_period, self.transmit_tick)
            self.transmit_ticker.start()

    def receive_mc_packet(self, key, payload):
        """Callback for when a multicast packet has been received.
        """
        # Inform the IO handler that a multicast packet has been received
        self.io.receive_mc_packet(key, payload)

    @stop_on_keyboard_interrupt
    def receive_tick(self):
        """Listen for packets and call :py:func:`receive_mc_packet` when
        received."""
        # Ask the protocol to listen for packet(s)
        self.receive_tick_inner()

        # Schedule this function to run again
        if not self.stop_now:
            self.receive_ticker = threading.Timer(
                self.rx_period, self.receive_tick)
            self.receive_ticker.start()

    def send_mc_packet(self, key, payload):
        """Transmit a multicast packet into the system given the appropriate
        key and payload.

        .. note::
            If this function does not respond quickly, the internal queue will
            fill up.
        """
        raise NotImplementedError

    def receive_tick_inner(self):
        """Listen for packets and call :py:func:`receive_mc_packet` when
        received."""
        raise NotImplementedError


class NSTSpiNNlinkProtocol(GenericUARTProtocol):
    def __init__(self, dev):
        super(NSTSpiNNlinkProtocol, self).__init__(dev)
        # AM: I have no idea if these values are even slightly sensible...
        self.tx_period = 0.00001
        self.rx_period = 0.00001

        # Set up the serial link
        self.serial = serial.Serial(dev, baudrate=8000000, rtscts=True,
                                    timeout=0.1)
        self.serial.write("S+\n")  # Send SpiNNaker packets to host

    def send_mc_packet(self, key, payload):
        """Transmit a multicast with the given key and payload into the system.
        """
        msg = "%08x.%08x\n" % (key, payload)
        self.serial.write(msg)
        self.serial.flush()

    def receive_tick_inner(self):
        """Listen for packets and call :py:func:`receive_mc_packet` when an MC
        packet is received.
        """
        try:
            data = self.serial.readline()

            if '.' in data:
                parts = [int(p, 16) for p in data.split('.')]
                if len(parts) == 3:
                    (header, key, payload) = parts
                    self.receive_mc_packet(key, payload)
        except IOError:  # No data to read
            pass


class SpIOUARTProtocol(GenericUARTProtocol):
    def __init__(self, port=None, baudrate=3000000):
        # AM: I have no idea if these values are even slightly sensible...
        self.tx_period = 0.00001
        self.rx_period = 0.00001

        self.packet_struct = struct.Struct("<LL")

        self.serial = serial.Serial(port, baudrate=baudrate,
                                    rtscts=True, timeout=1.0)
        self.spio_uart_sync()

        super(SpIOUARTProtocol, self).__init__()

    def spio_uart_sync(self):
        """Send a sync sequence to the remote device.

        XXX: Currently will not be safe to call once the send_mc_packet and
        receive_mc_packet functions are being executed.
        """
        # Send sync sequence
        self.serial.write("\x00"*13 + "\xFF")

        # Receive a sync sequence
        # Note: This is not a strict/robust check but it is sufficient for the
        # general case.
        while True:
            zero_count = 0
            while zero_count < 5:
                char = self.serial.read(1)
                if char == "\x00":
                    zero_count += 1
                else:
                    zero_count = 0
            while char == "\x00":
                char = self.serial.read(1)
            if char == "\xFF":
                break

    def send_mc_packet(self, key, payload):
        """Transmit a multicast with the given key and payload into the system.
        """
        # A multicast packet with a payload
        pkt = "\x02"
        pkt += self.packet_struct.pack(key, payload)

        # Calculate the checksum
        checksum = 0x00
        for b in pkt:
            checksum ^= ord(b)
        checksum ^= checksum >> 4
        checksum ^= checksum >> 2
        checksum ^= checksum >> 1
        checksum = (checksum & 1) ^ 1

        # Add checksum to header
        pkt = chr(ord(pkt[0]) | checksum) + pkt[1:]

        # Send
        self.serial.write(pkt)

    def receive_tick_inner(self):
        """Listen for packets and call :py:func:`receive_mc_packet` when an MC
        packet is received.
        """
        head_chr = self.serial.read(1)
        if not head_chr:
            # Terminate early on timeout
            return

        head = ord(head_chr)
        is_multicast = head & 0xC0 == 0x00
        long_packet = head & 0x02 == 0x02

        if is_multicast and long_packet:
            # Grab the key & payload
            key, payload = self.packet_struct.unpack(self.serial.read(8))

            # XXX: No parity checks are carried out
            self.receive_mc_packet(key, payload)
        else:
            # Ignore non multicast packets or multicast packets without
            # payloads
            self.serial.read(8 if long_packet else 4)
