import mock
import socket
import time
import pytest

from ..sdp_reactor import SDPReactor
from ..packets import SDPPacket, SDPFilter


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
    sock.bind(("", 17893))
    sock.settimeout(100)
    return sock


@pytest.fixture(scope='function')
def reactor(request):
    # Create and start a new reactor that will be called when finished
    reactor = SDPReactor(listen=17892)

    def stop():
        # Stop the reactor
        reactor.stop()

    reactor.start()
    request.addfinalizer(stop)
    return reactor


def test_receive(outsock, reactor):
    """Register a receipt callback and test that the SDPReactor listens for
    packets.
    """
    # Instantiate a new SDPReactor, register a callback with it.
    callback = mock.Mock(wraps=lambda p: None)
    reactor.register_received_callback(callback, SDPFilter(tag=0))

    # Transmit a new SDP packet (with iptag=1) to the reactor, the callback
    # should not be called.
    packet = SDPPacket(False, 1, 0, 0, 0, 0, 0, 0, 0, 0, b'')
    outsock.sendto(packet.bytestring, ("127.0.0.1", 17892))
    time.sleep(0.05)
    assert not callback.called

    # Transmit a new SDP packet (with iptag=1) to the reactor, the callback
    # should be called.
    packet = SDPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0, b'')
    outsock.sendto(packet.bytestring, ("127.0.0.1", 17892))
    while not callback.called:
        time.sleep(0.01)

    # Check that the callback was called with an appropriate packet
    r_packet = callback.call_args[0][0]
    assert isinstance(r_packet, SDPPacket)
    assert r_packet.bytestring == packet.bytestring


def test_transmit(reactor, insock):
    """Test that the SDPReactor transmits packets as directed.
    """
    # Create a transmitted callback, transmit a packet and check that it is
    # received.
    packet = SDPPacket(False, 0, 0, 0, 1, 1, 0, 0, 1, 1, b'\xFE\xCA')
    transmitted_callback = mock.Mock()

    reactor.transmit(packet, "127.0.0.1", 17893, transmitted_callback)
    sent_waits = 0
    while not transmitted_callback.called:
        time.sleep(0.005)  # Wait until the packet has been transmitted
        sent_waits += 1
        
        assert sent_waits < 10, "Waited too long for packet to be sent."

    # Now check that we can receive the packet
    data = insock.recv(512)
    assert data == packet.bytestring
