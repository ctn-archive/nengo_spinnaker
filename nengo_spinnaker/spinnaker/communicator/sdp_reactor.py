"""Communication primitives for the SDP protocol."""
import threading
from . import packets


class SDPReactor(threading.Thread):
    """Transmits and receives SDP formatted UDP messages.

    The SDPReactor manages transmitting and receiving SDP formatted UDP
    messages to and from SpiNNaker machines.  Users can request that certain
    callbacks be used when packets are transmitted or received.  On
    top of this layer of abstraction one can build flow-control or data
    transformation tools, e.g., for implementing flow-controlled SCP.

    Transmitting SDP Messages is trivial, once the reactor is running the
    :py:func:`~SDPReactor.transmit` method may be called with a byte-string and
    a destination: the bytestring is taken to be the complete packet.

    To receive packets a receive callback may be registered.  Callbacks may be
    registered with filters such that they are only called when packets are
    received on specific ports or from specific IP addresses.  Every callback
    that matches a received packet will be called.  Callbacks act to block the
    reactor and should be relatively light weight.
    """
    def __init__(self, out_ports=list()):
        """Create a new SDPReactor.

        Notes
        -----
        """
        raise NotImplementedError

    def transmit(self, packet, address, port):
        """Send a SDP packet over UDP.

        Parameters
        ----------
        packet : :py:class:`~.packets.SDPPacket`
            An SDP packet to transmit over UDP.
        address : string
            The address to which the data should be transmitted.
        port : int
            The port from which this packet should be sent.

        Notes
        -----
        It is not guaranteed that the packet will be transmitted immediately.
        Packets are added to a queue which is processed by the reactor.  This
        call may be modified to be blocking as and when required.
        """
        raise NotImplementedError

    def register_received_callback(self, callback, filter=SDPFilter()):
        """Register a function which should be called when a packet is
        received.

        Parameters
        ----------
        callback : method
            A function which should be called when an appropriate packet has
            been received.  The packets which will trigger the callback are
            defined by the filter argument.  The callback should accept an
            :py:class`~.packets.SDPPacket`.
        filter : :py:class:`~SDPFilter`
            Packets which match the filter will cause the callback to be
            called.

        Notes
        -----
        Calling a callback blocks the reactor thread.  Any callbacks should be
        light-weight.
        """
        raise NotImplementedError
