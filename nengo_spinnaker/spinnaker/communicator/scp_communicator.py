"""A stop-and-wait blocking implementation of the SCP protocol.
"""
import collections
import socket
from . import packets


CoreInfo = collections.namedtuple(
    'CoreInfo', "p2p_address physical_cpu virt_cpu version "
                "buffer_size build_date version_string")


class SCPTimeoutError(IOError):
    """Raised when an SCP is not acknowledged within the given period of time.
    """
    pass


class SCPCommunicator(object):
    """Implements the SCP protocol for communicating with a SpiNNaker machine.
    """
    def __init__(self, spinnaker_host, n_tries=5, timeout=0.5):
        """Create a new communicator to handle control of the given SpiNNaker
        host.

        Parameters
        ----------
        spinnaker_host : str
            A IP address or hostname of the SpiNNaker machine to control.
        n_tries : int
            The maximum number of tries to communicate with the machine before
            failing.
        timeout : float
            The timeout to use on the socket.
        """
        # Create a socket to communicate with the SpiNNaker machine
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.sock.connect((spinnaker_host, 17893))

        # Store the number of tries that will be allowed
        self.n_tries = n_tries

        # The current seq value
        self._seq = 0

    def _send_scp(self, x, y, p, cmd, arg1=0, arg2=0, arg3=0, data=b''):
        """Transmit a packet to the SpiNNaker machine and block until an
        acknowledgement is received.

        Returns
        -------
        packet : SCPPacket
            The packet that was received in acknowledgement of the transmitted
            packet.
        """
        # Construct the packet that will be sent
        packet = packets.SCPPacket(
            reply_expected=True, tag=0xff, dest_port=0, dest_cpu=p,
            src_port=7, src_cpu=31, dest_x=x, dest_y=y, src_x=0, src_y=0,
            cmd_rc=cmd, seq=self._seq, arg1=arg1, arg2=arg2, arg3=arg3,
            data=data
        )

        # Repeat until a reply is received or we run out of tries.
        n_tries = 0
        while n_tries < self.n_tries:
            # Transit the packet
            self.sock.send(b'\x00\x00' + packet.bytestring)
            n_tries += 1

            try:
                # Try to receive the returned acknowledgement
                ack = self.sock.recv(512)
            except IOError:
                # There was nothing to receive from the socket
                continue

            # Convert the possible returned packet into a SDPPacket and hence
            # to an SCPPacket.  If the seq field matches the expected seq then
            # the acknowledgement has been returned.
            # Unsure about the padding here
            scp = packets.SCPPacket.from_sdp_packet(
                packets.SDPPacket.from_bytestring(ack[2:]))
            
            if scp.seq == self._seq:
                # The packet is the acknowledgement.  Increment the sequence
                # indicator and return the packet.
                self._seq ^= 1
                return scp

        # The packet we transmitted wasn't acknowledged.
        raise SCPTimeoutError(
            "Exceeded {} tries when trying to transmit packet.".format(
                self.n_tries)
        )

    def software_version(self, x, y, p):
        """Get the software version for a given SpiNNaker core.
        """
        sver = self._send_scp(x, y, p, 0)

        # Format the result
        # arg1 => p2p address, physical cpu, virtual cpu
        p2p = sver.arg1 >> 16
        pcpu = (sver.arg1 >> 8) & 0xff
        vcpu = sver.arg1 & 0xff

        # arg2 => version number and buffer size
        version = (sver.arg2 >> 16) / 100.
        buffer_size = (sver.arg2 & 0xffff)

        return CoreInfo(p2p, pcpu, vcpu, version, buffer_size, sver.arg3,
                        sver.data)
