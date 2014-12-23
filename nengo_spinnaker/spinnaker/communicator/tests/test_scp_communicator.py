import mock
import threading
import time

from .. import packets
from ..scp_communicator import SCPReactor


class TestSCPReactor(object):
    """Test the SCPReactors implementation of the API and flow-control.
    """
    def test_transmit(self):
        """Test that the SCPReactor can successfully transmit SCP packets.
        Check that (for stop-and-wait flow-control) NO further (new) SCP
        packets are transmitted until the packet is acknowledged.
        """
        # Create a mock SDPReactor which we can control from this thread.
        sdp_reactor = mock.Mock(spec_set=['transmit',
                                          'register_received_callback'])

        # Create the SCPReactor using the SDPReactor, check that the SCPReactor
        # registers a somewhat sensible callback.
        scp_reactor = SCPReactor(sdp_reactor, "127.0.0.1")
        assert sdp_reactor.register_received_callback.called
        call_args = sdp_reactor.register_received_callback.call_args[0]
        assert call_args[0] == scp_reactor.receive_packet
        rx_callback = call_args[0]
        assert isinstance(call_args[1], packets.SDPFilter)
        assert (call_args[1].fields['tag'](0) and
                not call_args[1].fields['tag'](1))  # Listen for IPtag=0
        assert len(call_args[1].fields) == 1

        # Transmit the SVER command several times, assert that is is only
        # transmitted once during this short period of time.
        success = mock.Mock()
        failure = mock.Mock()
        for p in range(7):
            scp_reactor.transmit(0, 0, p, 0, n_args_expected=0,
                                 on_success=success, on_failure=failure)

        assert sdp_reactor.transmit.call_count == 1  # Only 1 packet sent out

        # Check that the packet is sensible
        assert len(sdp_reactor.transmit.call_args[0]) == 4
        sent_packet = sdp_reactor.transmit.call_args[0][0]
        expected_packet = packets.SCPPacket(
            True, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, b'')
        assert sent_packet.bytestring == expected_packet.bytestring
        assert sdp_reactor.transmit.call_args[0][1] == "127.0.0.1"
        assert sdp_reactor.transmit.call_args[0][2] == 17892
        transmit_callback = sdp_reactor.transmit.call_args[0][3]

        # Check that no tries have been recorded
        assert scp_reactor.n_tries == 0

        # Simulate transmitting the packet and ensure that the number of tries
        # is incremented.
        transmit_callback()
        assert scp_reactor.n_tries == 1

        # Ensure that after a reasonable pause the packet is retransmitted.
        wait = 0
        while scp_reactor.n_tries < 5:
            # Mock the SDP reactor transmitting the packet
            transmit_callback = sdp_reactor.transmit.call_args[0][3]
            transmit_callback()

            # Sleep to give the SCP reactor time to act
            time.sleep(.1)
            assert wait < 10, "SCPReactor failed to resend last packet."
            wait += 1

        # Ensure that the failure handler was called
        assert failure.called

        # Clear the queue and transmit another packet, this time acknowledge it
        # and ensure that the number of tries doesn't increment and that
        # success and not failure is indicated.
        ack_packet = packets.SDPPacket(False, 0, 7, 31, 0, 0, 0, 0, 0, 0,
                                       12*'b\x00')

        def success_f(packet):
            assert isinstance(packet, packets.SCPPacket)
            assert packet.arg1 is None
            assert packet.arg2 is None
            assert packet.arg3 is None
            assert packet.bytestring == ack_packet.bytestring

        failure = mock.Mock()
        success = mock.Mock(wraps=success_f)
        scp_reactor.queue = list()
        sdp_reactor.transmit.side_effect = lambda p, a, po, c : c()
        scp_reactor.transmit(0, 0, 0, 0, n_args_expected=0,
                             on_success=success, on_failure=failure)

        # Acknowledge by calling the SCP reactor's receive callback.
        rx_callback(ack_packet)

        # Sleep to ensure fail isn't called!
        time.sleep(.5)
        assert scp_reactor.n_tries == 1
        assert not failure.called
        assert success.called
