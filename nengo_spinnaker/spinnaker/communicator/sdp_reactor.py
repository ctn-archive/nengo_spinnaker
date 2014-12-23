"""Communication primitives for the SDP protocol."""
import collections
from six.moves import queue
import socket
import threading
import time
from . import packets


QueuedPacket = collections.namedtuple('QueuedPacket',
                                      'packet address port callback')


class SDPReactor(threading.Thread):
    """Transmits and receives SDP formatted UDP messages.

    The SDPReactor manages transmitting and receiving SDP formatted UDP
    messages to and from SpiNNaker machines.  Users can request that certain
    callbacks be used when packets are transmitted or received.  On
    top of this layer of abstraction one can build flow-control or data
    transformation tools, e.g., for implementing flow-controlled SCP.

    Transmitting SDP Messages is trivial, once the reactor is running the
    :py:func:`~SDPReactor.transmit` method may be called with an SDP packet and
    a destination.

    To receive packets a receive callback may be registered.  Callbacks may be
    registered with filters such that they are only called when packets with
    certain characteristics are received.  Every callback that matches a
    received packet will be called.  Callbacks act to block the reactor and
    should be relatively light weight.

    Notes
    -----
    At some later point the SDPReactor will be registered to talk to a given
    SpiNNaker machine and may be given a map of what ranges of co-ordinates map
    to which IP address and port.  For the moment, however, the user is
    expected to know where they wish to send their packets.
    """
    def __init__(self, listen=17892):
        """Create a new SDPReactor."""
        # Prepare the thread
        super(SDPReactor, self).__init__(name="SDPReactor")
        self._halt = False

        # Create the listening socket (non-blocking)
        self.in_sock = socket.socket(type=socket.SOCK_DGRAM)
        self.in_sock.bind(("", listen))
        self.in_sock.setblocking(0)

        # Create the transmitting socket (blocking)
        self.out_sock = socket.socket(type=socket.SOCK_DGRAM)

        # Create the outgoing queue
        self.out_queue = queue.Queue()

        # Create the list of callbacks
        self._callbacks = list()

    def transmit(self, packet, address, port, transmitted_callback=None):
        """Send a SDP packet over UDP.

        Parameters
        ----------
        packet : :py:class:`~.packets.SDPPacket`
            An SDP packet to transmit over UDP.
        address : string
            The address to which the data should be transmitted.
        port : int
            The port to which this packet should be sent.
        transmitted_callback : callable
            A callback which will be called once this packet has been
            transmitted.

        Notes
        -----
        It is not guaranteed that the packet will be transmitted immediately.
        Packets are added to a queue which is processed by the reactor.  This
        call may be modified to be blocking as and when required.
        """
        self.out_queue.put(
            QueuedPacket(packet, address, port, transmitted_callback))

    def register_received_callback(self, callback, sdp_filter=None):
        """Register a function which should be called when a packet is
        received.

        Parameters
        ----------
        callback : method
            A function which should be called when an appropriate packet has
            been received.  The packets which will trigger the callback are
            defined by the filter argument.  The callback should accept an
            :py:class`~.packets.SDPPacket`.
        sdp_filter : :py:class:`~SDPFilter`
            Packets which match the filter will cause the callback to be
            called.

        Notes
        -----
        Calling a callback blocks the reactor thread.  Any callbacks should be
        light-weight.
        """
        self._callbacks.append((sdp_filter, callback))

    def stop(self):
        """Terminate the thread."""
        self._halt = True
        self.join()

    def run(self):
        # While not halted try to get data from the socket, then try to send,
        # then sleep.
        while not self._halt:
            incoming = None
            try:
                incoming = self.in_sock.recv(512)
            except IOError:
                # There was no data to receive
                pass

            # If there was received data, then interpret it as an SDP packet
            if incoming is not None:
                packet = packets.SDPPacket.from_bytestring(incoming)

                # Call all required callbacks
                for (f, callback) in self._callbacks:
                    if f(packet):
                        callback(packet)

            # Send a packet if the queue isn't empty
            if not self.out_queue.empty():
                try:
                    qpacket = self.out_queue.get(block=False)

                    # Send the packet
                    self.out_sock.sendto(qpacket.packet.bytestring,
                                         (qpacket.address, qpacket.port))

                    # Call the callback
                    qpacket.callback()
                except queue.Empty:
                    # The queue was empty, nothing to send
                    pass

            # Sleep to give other threads a chance.
            time.sleep(0.0001)
