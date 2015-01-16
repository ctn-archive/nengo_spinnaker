"""Tests for the stop-and-wait SCP implementation.
"""
import pytest
import socket
import time
import threading

from ..packets import SCPPacket, SDPPacket


PRETEND_SPINNAKER_PORT = 51234


@pytest.fixture(scope='module')
def outsock():
    # Create a new socket
    sock = socket.socket(type=socket.SOCK_DGRAM)
    sock.setblocking(0)
    return sock


@pytest.fixture(scope='module')
def insock():
    # Create a new socket
    sock = socket.socket(type=socket.SOCK_DGRAM)
    sock.bind(("", PRETEND_SPINNAKER_PORT))
    sock.settimeout(.1)
    return sock


class Listener(threading.Thread):
    """A thread which will listen for SCP packets.
    """
    def __init__(self, socket, received_callback):
        super(Listener, self).__init__()
        self.socket = socket
        self._stop = False
        self._received_callback = received_callback

    def stop(self):
        self._stop = True
        self.join()

    def run(self):
        while not self._stop:
            # Run until the calling thread halts
            try:
                data = self.socket.recv(512)
            except IOError:
                # There was no data for us to receive
                continue

            # Convert the SDP packet into an SCP packet and call the callback
            self._received_callback(
                SCPPacket.from_sdp_packet(SDPPacket.from_bytestring(data)))


class SeqCounter(object):
    def __init__(self, seq, ack=False):
        self.count = 0
        self.seq = seq

    def __call__(self, scp_packet):
        assert isinstance(scp_packet, SCPPacket)
        if scp_packet.seq == self.seq:
            self.count += 1


def test_transmit(insock, outsock):
    """Test that an SCP packet can be correctly sent and is not retried if a
    response occurs within the timeout.
    """
    # Create a new listener with a callback which will immediately acknowledge
    # the transmitted packet.
    class AckAndCount(object):
        def __init__(self, expected, ack):
            self.count = 0
            self.ack = ack
            assert isinstance(expected, SCPPacket)
            self.expected = expected

        def __call__(self, scp_packet):
            # Check the packet was received successfully
            assert isinstance(scp_packet, SCPPacket)
            assert scp_packet.bytestring == self.expected.bytestring
            self.count += 1

            # Acknowledge the packet

    # Transmit a new SCP packet and ensure that it is received
    packet = SCPPacket(
        reply_expected=True, tag=0, dest_port=0, dest_cpu=0,
        src_port=7, src_cpu=31, dest_x=0, dest_y=0, src_x=0, src_y=0,
        cmd_rc=128, seq=0x0, arg1=0, arg2=1, arg3=3, data=b''
    )
    ack = SCPPacket(
        reply_expected=True, tag=0, dest_port=7, dest_cpu=31,
        src_port=0x0, src_cpu=0x0, dest_x=0, dest_y=0, src_x=0, src_y=0,
        cmd_rc=128, seq=0x0, arg1=0, arg2=1, arg3=3, data=b''
    )

    # Start the listener
    try:
        acker = AckAndCount(packet, ack)
        lst = Listener(insock, acker)
        lst.start()

        # Transmit the packet
        outsock.sendto(packet.bytestring, ('localhost', 17893))
    finally:
        # Assert that only one was received, (i.e., no repeats)
        time.sleep(1.)
        lst.stop()
        assert acker.count == 1, "Incorrect number of packets received"


def test_transmit_resend(insock, outsock):
    """Test that an SCP packet is resent if not acknowledged and that an error
    is raised if it is not acknowledged within 5 retries.
    """
    class Count(object):
        def __init__(self, expected):
            self.count = 0
            self.expected = expected

        def __call__(self, scp_packet):
            # Check the packet was received successfully
            assert isinstance(scp_packet, SCPPacket)
            assert scp_packet.bytestring == self.expected.bytestring
            self.count += 1

    # Transmit a new SCP packet and ensure that it is received
    packet = SCPPacket(
        reply_expected=True, tag=0, dest_port=0, dest_cpu=0,
        src_port=7, src_cpu=31, dest_x=0, dest_y=0, src_x=0, src_y=0,
        cmd_rc=128, seq=0x0, arg1=0, arg2=1, arg3=3, data=b''
    )

    # Start the listener
    try:
        acker = AckAndCount(packet, ack)
        lst = Listener(insock, acker)
        lst.start()

        # Transmit the packet
        with pytest.raises():
            outsock.sendto(packet.bytestring, ('localhost', 17893))
    finally:
        # Assert that only one was received, (i.e., no repeats)
        time.sleep(1.)
        lst.stop()
        assert acker.count == 5, "Incorrect number of packets received"
