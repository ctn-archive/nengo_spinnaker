import pytest

from ..scp_communicator import SCPCommunicator


@pytest.mark.spinnaker
def test_sver(spinnaker_ip):
    """Test getting the software version data."""
    # Create an SCPCommunicator for the given SpiNNaker IP address.
    comms = SCPCommunicator(spinnaker_ip)

    # (Assuming a 4-node board) Get the software version for a number of cores.
    for x in range(2):
        for y in range(2):
            for p in range(16):
                sver = comms.software_version(x, y, p)
                assert sver.virt_cpu == p
                assert "SpiNNaker" in sver.version_string
                assert sver.version >= 1.3
