"""SpiNNaker machine control using the SCP protocol with flow control.
"""
from .packets import SCPPacket


class SCPReactor(object):
    """Transmits and receives SCP packets, with flow control.

    The SCPReactor piggy-backs on top of the
    :py:class`~.sdp_reactor.SDPReactor` to transmit and receive SCP packets.
    Like the SDPReactor it allows simply for transmitting and receiving
    packets, with received packets being used to drive callbacks.  Unlike the
    SDP reactor it implements flow control using the `seq` field of the SCP
    packets.
    """
    def __init__(self, sdp_reactor):
        """Create a new SCP reactor.

        Parameters
        ----------
        sdp_reactor : :py:class:`~.sdp_reactor.SDPReactor`
            A SDP reactor which the SCP reactor can register itself with to
            transmit and receive packets.
        """
        raise NotImplementedError

    def transmit(self, packet, on_success=None, on_failure=None):
        """Send a SCP packet to the machine.

        Parameters
        ----------
        packet : :py:class:`~.packets.SCPPacket`
            An SCP packet to transmit to the machine.
        on_success : callable
            A callback that will be called when the packet has been transmitted
            and acknowledged by the SpiNNaker machine.  The callback should
            accept the packet that was transmitted.
        on_failure : callable
            Callback that will be called when the packet has not been
            successfully sent after several retries.  Note that failure
            indicates either an outgoing or an incoming issue (cannot send, or
            cannot receive ACKS).

        Notes
        -----
        Calling a callback blocks the reactor thread.  Any callbacks should be
        light-weight.
        """
        raise NotImplementedError

    def register_received_callback(self, callback, filter=None):
        """Register a function which should be called when a packet is
        received.

        Parameters
        ----------
        callback : callable
            A callable which should be called when an appropriate packet has
            been received.  The packets which will trigger the callback are
            defined by the filter argument.  The callback should accept a
            :py:class:`~.packets.SCPPacket`.
        filter :
            Packets which match the filter will cause the callback to be
            called.

        Notes
        -----
        Calling a callback blocks the reactor thread.  Any callbacks should be
        light-weight.
        """
        raise NotImplementedError


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
