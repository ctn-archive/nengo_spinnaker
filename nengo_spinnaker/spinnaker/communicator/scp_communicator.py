"""SpiNNaker machine control using the SCP protocol with flow control.
"""
import collections
import logging
import threading

from .packets import SCPPacket, SDPFilter

logger = logging.getLogger()


QueuedPacket = collections.namedtuple(
    'QueuedPacket', 'packet on_success on_failure n_args_exp')


class SCPReactor(object):
    """Transmits and receives SCP packets, with flow control.

    The SCPReactor piggy-backs on top of the
    :py:class`~.sdp_reactor.SDPReactor` to transmit and receive SCP packets.
    Like the SDPReactor it allows simply for transmitting and receiving
    packets, with received packets being used to drive callbacks.  Unlike the
    SDP reactor it implements flow control using the `seq` field of the SCP
    packets, in this implementation the flow-control protocol used is
    Stop-and-Wait; it is planned to migrate to sliding window flow-control as
    soon as possible.
    """
    def __init__(self, sdp_reactor, address, port=17892):
        """Create a new SCP reactor.

        Parameters
        ----------
        address : string
            The IP address this SCP reactor communicates with. (Temporary until
            discovery is made possible).
        sdp_reactor : :py:class:`~.sdp_reactor.SDPReactor`
            A SDP reactor which the SCP reactor can register itself with to
            transmit and receive packets.
        """
        # Register the SCP reactor with the SDP reactor
        sdp_reactor.register_received_callback(self.receive_packet,
                                               SDPFilter(tag=0))
        self.sdp_reactor = sdp_reactor

        # Store the address to send to
        self.address = address
        self.port = port

        # Store the current Seq value
        self.seq = 0

        # Store the current packet and number of tries, also a queue of packets
        # to send.
        self.sent_packet = None
        self.retransmit_timer = None
        self.n_tries = 0
        self.queue = list()

    def transmit(self, x, y, p, cmd_rc, arg1=None, arg2=None, arg3=None,
                 data=b'', n_args_expected=3, on_success=None,
                 on_failure=None):
        """Send a SCP packet to the machine.

        Parameters
        ----------
        x : int
            The x co-ordinate of the processor to transmit the message to.
        y : int
            The y co-ordinate of the processor to transmit the message to.
        p : int
            Index of the virtual CPU to which the message should be
            transmitted.
        cmd_rc : int
            The command code to transmit.
        arg1 : int or None
        arg2 : int or None
        arg3 : int or None
        data : bytestring
            A bytestring of data to transmit in the SCP packet, this will be
            placed after the SCP header.
        n_args_expected : int
            The number of argument fields expected in the reply/acknowledgement
            from the SpiNNaker machine.
        on_success : callable
            A callback that will be called when the packet has been transmitted
            and acknowledged by the SpiNNaker machine.  The callback should
            accept the packet that was returned from the SpiNNaker machine.
        on_failure : callable
            Callback that will be called when the packet has not been
            successfully sent after several retries.  Note that failure
            indicates either an outgoing or an incoming issue (cannot send, or
            cannot receive ACKS).  This callback should accept no arguments.

        Notes
        -----
        Calling a callback blocks the reactor thread.  Any callbacks should be
        light-weight.
        """
        # Construct a new SCP packet
        packet = SCPPacket(True, 0, 0, p, 0, 0, x, y, 0, 0,
                           cmd_rc, 0xFFFF, arg1, arg2, arg3, data)

        # Add this packet to the queue and then transmit if possible.
        self.queue.append(QueuedPacket(packet, on_success, on_failure,
                                       n_args_expected))
        self.transmit_packet()

    def transmit_packet(self):
        """Transmit a packet from the queue if there it is safe to transmit."""
        # For Stop-and-Wait we can only transmit a new packet if there is no
        # packet waiting.
        if self.sent_packet is None:
            if len(self.queue) < 0:
                return  # No packets to send

            # Send the first packet in the queue
            qpacket = self.queue.pop(0)

            # Swap out the seq value
            qpacket.packet.seq = self.seq

            # Transmit the packet
            self.sent_packet = qpacket
            self.n_tries = 0
            self.sdp_reactor.transmit(
                qpacket.packet, self.address, self.port,
                self.make_packet_transmitted_callback(self.seq)
            )

    def retransmit_packet(self):
        """Retransmit the last packet as transmitting it failed."""
        self.sdp_reactor.transmit(
            self.sent_packet.packet, self.address, self.port,
            self.make_packet_transmitted_callback(self.seq)
        )

    def make_packet_transmitted_callback(self, seq):
        """Make a callback for a transmitted packet."""
        def callback():
            # Increment the number of tries
            self.n_tries += 1

            # Create a time-out after which we'll try to resend again, or will
            # fail.
            if self.n_tries < 5:
                self.retransmit_timer = threading.Timer(
                    0.05, self.retransmit_packet)
            else:
                self.retransmit_timer = threading.Timer(0.05, self.fail_packet)
            self.retransmit_timer.start()

        return callback

    def fail_packet(self):
        """Fails the last sent packet and stalls the queue."""
        logger.warning(
            "Failed to transmit packet after {} retries.".format(self.n_tries))
        self.sent_packet.on_failure()
        self.sent_packet = None
        self.retransmit_timer = None
        self.n_tries

    def receive_packet(self, sdp_packet):
        """Receive a packet from the SDP reactor.

        Will inspect the `seq` field to determine which callback needs to be
        called.  If there are any packets awaiting transmission this will
        trigger transmission of the first.

        Notes
        -----
        This is part of the internal API.
        """
        # Clear the retransmission timer
        self.retransmit_timer.cancel()

        # Call the success indicator with this packet as an SCP packet.
        self.sent_packet.on_success(
            SCPPacket.from_sdp_packet(sdp_packet,
                                      n_args=self.sent_packet.n_args_exp)
        )


class SCPCommunicator(object):
    """Handles control of a SpiNNaker machine using the SCP protocol.
    """
    def open_sdram(x, y):
        """Open the file-like object representing the SDRAM of a SpiNNaker
        chip.

        Parameters
        ----------
        x : int
            The x-coordinate of the chip to open the SDRAM of.
        y : int
            The y-coordinate of the chip to open the SDRAM of.

        Returns
        -------
        :py:class:`~SDRAMFileObject`
            An file-like object which allows for read and write access of a
            chip's SDRAM.
        """
        raise NotImplementedError


class SDRAMFileObject(object):
    """A file-like object allowing access to SDRAM.
    """
    __slots__ = ['_pos', '_read_wait']

    def __init__(self, scp_reactor, x, y):
        """Create a new file-like object representing the SDRAM of a single
        SpiNNaker chip.

        This object will usually be initiated by an open call to the SCP
        reactor.

        Parameters
        ----------
        scp_reactor : SCPReactor
            A SCP reactor which provides flow control transmission and
            receiving of SCP messages.
        x : int
            The x-coordinate of the SDRAM to open.
        y : int
            The y-coordinate of the SDRAM to open.
        """
        raise NotImplementedError

    def close(self):
        """Close the SDRAM file-like object.

        This is a blocking call that will close the SDRAM file when all writes
        (and deferred reads) have been completed.
        """
        raise NotImplementedError

    def flush(self):
        """Flush writes on the SDRAM file-like object.

        This is a blocking call that will clear as soon as all writes to the
        SDRAM have been completed.
        """
        raise NotImplementedError

    def read(self, size=None):
        """Read at most size bytes from the SDRAM.

        If the `size` argument is missing or negative then all data until the
        end of the SDRAM will be read.  This is a blocking call and will
        complete when as many bytes as possible (the number specified or fewer
        if EOF is encountered) have been read.  The returned bytes will be in
        the form of a byte string.

        Parameters
        ----------
        size : int
            The number of bytes to read.

        Returns
        -------
        string
            A string of `size` or fewer bytes read from SDRAM.

        Raises
        ------
        IOError
            If the read fails due to network failure or similar.
        """
        raise NotImplementedError

    def seek(self, offset, whence):
        """Set the file's current position.
        """
        raise NotImplementedError

    def write(self, str):
        """Write a string to the SDRAM.

        This is a non-blocking call with no return value.  The string may not
        be fully written until either :py:func:`close` or :py:func:`flush` are
        called.

        Raises
        ------
        IOError
            If the write moves beyond the bounds of SDRAM or fails for any
            other reason.
        """
        raise NotImplementedError
