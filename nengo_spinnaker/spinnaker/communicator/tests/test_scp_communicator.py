import pytest
import socket
import time

from ..scp_communicator import SCPCommunicator
from .. import scp_communicator


@pytest.fixture(scope='module')
def comms(spinnaker_ip):
    return SCPCommunicator(spinnaker_ip)


@pytest.mark.spinnaker
def test_sver(comms):
    """Test getting the software version data."""
    # (Assuming a 4-node board) Get the software version for a number of cores.
    for x in range(2):
        for y in range(2):
            for p in range(16):
                sver = comms.software_version(x, y, p)
                assert sver.virt_cpu == p
                assert "SpiNNaker" in sver.version_string
                assert sver.version >= 1.3


@pytest.mark.spinnaker
@pytest.mark.parametrize("action", [scp_communicator.LEDAction.ON,
                                    scp_communicator.LEDAction.OFF,
                                    scp_communicator.LEDAction.TOGGLE,
                                    scp_communicator.LEDAction.TOGGLE])
def test_set_led(comms, action):
    """Test getting the software version data."""
    # (Assuming a 4-node board)
    for x in range(2):
        for y in range(2):
            comms.set_led(x, y, 1, action)

    time.sleep(0.05)


@pytest.mark.spinnaker
def test_write_and_read(comms):
    """Test write and read capabilities by writing a string to SDRAM and then
    reading back in a different order.
    """
    data = b'Hello, SpiNNaker'

    # You put the data in
    comms.write(0, 0, 0, 0x70000000, data[0:4],
                scp_communicator.DataType.WORD)
    comms.write(0, 0, 0, 0x70000004, data[4:6],
                scp_communicator.DataType.SHORT)
    comms.write(0, 0, 0, 0x70000006, data[6:],
                scp_communicator.DataType.BYTE)

    # You take the data out
    assert comms.read(0, 0, 1, 0x70000000, 1,
                      scp_communicator.DataType.BYTE) == data[0]
    assert comms.read(0, 0, 1, 0x70000000, 2,
                      scp_communicator.DataType.SHORT) == data[0:2]
    assert comms.read(0, 0, 1, 0x70000000, 4,
                      scp_communicator.DataType.WORD) == data[0:4]

    # Read out the entire string
    assert comms.read(0, 0, 1, 0x70000000, len(data),
                      scp_communicator.DataType.BYTE) == data
    assert comms.read(0, 0, 1, 0x70000000, len(data),
                      scp_communicator.DataType.SHORT) == data
    assert comms.read(0, 0, 1, 0x70000000, len(data),
                      scp_communicator.DataType.WORD) == data


@pytest.mark.spinnaker
def test_set_get_clear_iptag(comms):
    # Get our address, then add a new IPTag pointing
    ip_addr = comms.sock.getsockname()[0]
    port = 1234
    iptag = 7

    # Set IPTag 7 with the parameters from above
    comms.iptag_set(0, 0, iptag, ip_addr, port)

    # Get the IPtag and check that it is as we set it
    ip_tag = comms.iptag_get(0, 0, iptag)
    assert ip_addr == ip_tag.addr
    assert port == ip_tag.port
    assert ip_tag.flags != 0

    # Clear the IPTag
    comms.iptag_clear(0, 0, iptag)

    # Check that it is empty by inspecting the flag
    ip_tag = comms.iptag_get(0, 0, iptag)
    assert ip_tag.flags == 0

@pytest.mark.spinnaker
def test_bad_packet_length(comms):
    """Test transmitting a packet with an incorrect length, this should raise
    an error.
    """
    with pytest.raises(scp_communicator.BadPacketLengthError):
        comms._send_scp(0, 0, 0, 0, None, None, None, b'')


@pytest.mark.spinnaker
def test_invalid_command(comms):
    """Test transmitting a packet with an invalid CMD raises an error."""
    # Create an SCPCommunicator for the given SpiNNaker IP address.
    with pytest.raises(scp_communicator.InvalidCommandError):
        comms._send_scp(0, 0, 0, 6)


"""
@pytest.mark.spinnaker
def test_invalid_argument(spinnaker_ip):
    # Create an SCPCommunicator for the given SpiNNaker IP address.
    comms = SCPCommunicator(spinnaker_ip)

    with pytest.raises(scp_communicator.InvalidArgsError):
        comms._send_scp(0, 0, 0, scp_communicator.SCPCommands.LED, 128)
"""
