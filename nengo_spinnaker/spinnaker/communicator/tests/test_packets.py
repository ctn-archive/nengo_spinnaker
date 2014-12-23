import pytest
from ..packets import SDPPacket, SCPPacket, SDPFilter
from .. import packets


class TestRangedIntAttribute(object):
    def test_min_exclusive(self):
        # Test values are correctly checked in the min value is excluded from
        # the valid range.
        class X(object):
            y = packets.RangedIntAttribute(0, 10, min_inclusive=False)

        x = X()
        with pytest.raises(ValueError):
            x.y = 0

        x.y = 1

    def test_max_inclusive(self):
        # Test values are correctly checked in the max value is included in the
        # valid range.
        class X(object):
            y = packets.RangedIntAttribute(0, 10, max_inclusive=True)

        x = X()
        x.y = 10

    def test_min_max_fail(self):
        # Test that a value error is raised on instantiation if the min/max
        # values are wrong.
        with pytest.raises(ValueError):
            class X(object):
                y = packets.RangedIntAttribute(100, 0)

    def test_type_fail(self):
        class X(object):
            y = packets.RangedIntAttribute(0, 100)

        x = X()
        with pytest.raises(TypeError):
            x.y = "Oops!"

    def test_allow_none(self):
        class X(object):
            y = packets.RangedIntAttribute(0, 10, allow_none=False)

        x = X()
        with pytest.raises(ValueError):
            x.y = None

        class Y(object):
            y = packets.RangedIntAttribute(0, 10, allow_none=True)

        y = Y()
        y.y = None


class TestSDPPacket(object):
    """Test SDPPacket representations."""
    def test_from_bytestring_to_bytestring(self):
        """Test creating a new SDPPacket from a bytestring."""
        # Create bytestring representing a packet with:
        #     flags: 0x87
        #     tag: 0xF0
        #     dest_port: 7 (max 3 bits)
        #     dest_cpu: 0x0F (max 5 bits)
        #     src_port: 7
        #     src_cpu: 0x0E
        #     dest_x: 0xA5
        #     dest_y: 0x5A
        #     src_x: 0x0F
        #     src_y: 0xF0
        #     data: 0xDEADBEEF
        packet = b'\x87\xf0\xef\xee\xa5\x5a\x0f\xf0\xDE\xAD\xBE\xEF'
        sdp_packet = SDPPacket.from_bytestring(packet)

        assert isinstance(sdp_packet, SDPPacket)
        assert sdp_packet.reply_expected
        assert sdp_packet.tag == 0xF0
        assert sdp_packet.dest_port == 7
        assert sdp_packet.dest_cpu == 0x0F
        assert sdp_packet.src_port == 7
        assert sdp_packet.src_cpu == 0x0E
        assert sdp_packet.dest_x == 0xA5
        assert sdp_packet.dest_y == 0x5A
        assert sdp_packet.src_x == 0x0F
        assert sdp_packet.src_y == 0xF0
        assert sdp_packet.data == b'\xDE\xAD\xBE\xEF'

        # Check that the bytestring this packet creates is the same as the one
        # we specified before.
        assert sdp_packet.bytestring == packet

    def test_from_bytestring_no_reply(self):
        """Test creating a new SDPPacket from a bytestring."""
        # Create bytestring representing a packet with:
        #     flags: 0x07
        packet = b'\x07\xf0\xef\xee\xa5\x5a\x0f\xf0\xDE\xAD\xBE\xEF'
        sdp_packet = SDPPacket.from_bytestring(packet)

        assert isinstance(sdp_packet, SDPPacket)
        assert not sdp_packet.reply_expected

    def test_values(self):
        """Check that errors are raised when values are out of range."""
        with pytest.raises(TypeError):  # Ints should be ints
            SDPPacket(False, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # IPTag is 8 bits
            SDPPacket(False, 300, 0, 0, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # IPTag is 8 bits
            SDPPacket(False, -1, 0, 0, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_port is 3 bits
            SDPPacket(False, 255, 8, 0, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_port is 3 bits
            SDPPacket(False, 255, -1, 0, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_cpu is 5 bits but should range 0..17
            SDPPacket(False, 255, 7, 18, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_cpu is 5 bits but should range 0..17
            SDPPacket(False, 255, 7, -1, 0, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # src_port is 3 bits
            SDPPacket(False, 255, 7, 17, 8, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # src_port is 3 bits
            SDPPacket(False, 255, 7, 17, -1, 0, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # src_cpu is 5 bits but should range 0..17
            SDPPacket(False, 255, 7, 17, 7, 18, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # src_cpu is 5 bits but should range 0..17
            SDPPacket(False, 255, 7, 17, 7, -1, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_x is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 256, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_x is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, -1, 0, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_y is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 255, 256, 0, 0, b'')

        with pytest.raises(ValueError):
            # dest_y is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 255, -1, 0, 0, b'')

        with pytest.raises(ValueError):
            # src_x is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 255, 255, 256, 0, b'')

        with pytest.raises(ValueError):
            # src_x is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 255, 255, -1, 0, b'')

        with pytest.raises(ValueError):
            # src_y is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 255, 255, 255, 256, b'')

        with pytest.raises(ValueError):
            # src_y is 8 bits
            SDPPacket(False, 255, 7, 17, 7, 17, 255, 255, 255, -1, b'')

        with pytest.raises(ValueError):
            # Data can only be 272 bytes long
            SDPPacket(False, 255, 7, 17, 7, 17, 255, 255, 255, 255,
                      273*b'\x00')


class TestSCPPacket(object):
    """Test packets conforming to the SCP protocol."""
    def test_from_bytestring_short(self):
        """Test creating an SCP Packet from a bytestring when the SCP Packet is
        short (no arguments, no data).
        """
        # Create bytestring representing a packet with:
        #     flags: 0x87
        #     tag: 0xF0
        #     dest_port: 7 (max 3 bits)
        #     dest_cpu: 0x0F (max 5 bits)
        #     src_port: 7
        #     src_cpu: 0x0E
        #     dest_x: 0xA5
        #     dest_y: 0x5A
        #     src_x: 0x0F
        #     src_y: 0xF0
        #     cmd_rc: 0xDEAD
        #     seq: 0xBEEF
        packet = b'\x87\xf0\xef\xee\xa5\x5a\x0f\xf0\xAD\xDE\xEF\xBE'
        sdp_packet = SDPPacket.from_bytestring(packet)
        scp_packet = SCPPacket.from_sdp_packet(sdp_packet, n_args=0)

        assert isinstance(scp_packet, SCPPacket)
        assert scp_packet.reply_expected
        assert scp_packet.tag == 0xF0
        assert scp_packet.dest_port == 7
        assert scp_packet.dest_cpu == 0x0F
        assert scp_packet.src_port == 7
        assert scp_packet.src_cpu == 0x0E
        assert scp_packet.dest_x == 0xA5
        assert scp_packet.dest_y == 0x5A
        assert scp_packet.src_x == 0x0F
        assert scp_packet.src_y == 0xF0
        assert scp_packet.cmd_rc == 0xDEAD
        assert scp_packet.seq == 0xBEEF
        assert scp_packet.arg1 == None
        assert scp_packet.arg2 == None
        assert scp_packet.arg3 == None
        assert scp_packet.data == b''

        # Check that the bytestring this packet creates is the same as the one
        # we specified before.
        assert scp_packet.bytestring == packet

    def test_from_bytestring(self):
        """Test creating a new SCPPacket from a bytestring."""
        # Create bytestring representing a packet with:
        #     flags: 0x87
        #     tag: 0xF0
        #     dest_port: 7 (max 3 bits)
        #     dest_cpu: 0x0F (max 5 bits)
        #     src_port: 7
        #     src_cpu: 0x0E
        #     dest_x: 0xA5
        #     dest_y: 0x5A
        #     src_x: 0x0F
        #     src_y: 0xF0
        #     cmd_rc: 0xDEAD
        #     seq: 0xBEEF
        #     arg1: 0xA5A5B7B7
        #     arg2: 0xCAFECAFE
        #     arg3: 0x5A5A7B7B
        #     data: 0xFEEDDEAF01
        packet = b'\x87\xf0\xef\xee\xa5\x5a\x0f\xf0\xAD\xDE\xEF\xBE' + \
                 b'\xB7\xB7\xA5\xA5\xFE\xCA\xFE\xCA\x7B\x7B\x5A\x5A' + \
                 b'\xFE\xED\xDE\xAF\x01'
        sdp_packet = SDPPacket.from_bytestring(packet)
        scp_packet = SCPPacket.from_sdp_packet(sdp_packet)

        assert isinstance(scp_packet, SCPPacket)
        assert scp_packet.reply_expected
        assert scp_packet.tag == 0xF0
        assert scp_packet.dest_port == 7
        assert scp_packet.dest_cpu == 0x0F
        assert scp_packet.src_port == 7
        assert scp_packet.src_cpu == 0x0E
        assert scp_packet.dest_x == 0xA5
        assert scp_packet.dest_y == 0x5A
        assert scp_packet.src_x == 0x0F
        assert scp_packet.src_y == 0xF0
        assert scp_packet.cmd_rc == 0xDEAD
        assert scp_packet.seq == 0xBEEF
        assert scp_packet.arg1 == 0xA5A5B7B7
        assert scp_packet.arg2 == 0xCAFECAFE
        assert scp_packet.arg3 == 0x5A5A7B7B
        assert scp_packet.data == b'\xFE\xED\xDE\xAF\x01'

        # Check that the bytestring this packet creates is the same as the one
        # we specified before.
        assert scp_packet.bytestring == packet

    def test_from_bytestring_0_args(self):
        """Test creating a new SCPPacket from a bytestring."""
        # Create bytestring representing a packet with:
        #     flags: 0x87
        #     tag: 0xF0
        #     dest_port: 7 (max 3 bits)
        #     dest_cpu: 0x0F (max 5 bits)
        #     src_port: 7
        #     src_cpu: 0x0E
        #     dest_x: 0xA5
        #     dest_y: 0x5A
        #     src_x: 0x0F
        #     src_y: 0xF0
        #     cmd_rc: 0xDEAD
        #     seq: 0xBEEF
        #     data: 0xA5A5B7B7CAFECAFE5A5A7B7BFEEDDEAF01
        packet = b'\x87\xf0\xef\xee\xa5\x5a\x0f\xf0\xAD\xDE\xEF\xBE' + \
                 b'\xB7\xB7\xA5\xA5\xFE\xCA\xFE\xCA\x7B\x7B\x5A\x5A' + \
                 b'\xFE\xED\xDE\xAF\x01'
        sdp_packet = SDPPacket.from_bytestring(packet)
        scp_packet = SCPPacket.from_sdp_packet(sdp_packet, n_args=0)

        assert scp_packet.cmd_rc == 0xDEAD
        assert scp_packet.seq == 0xBEEF
        assert scp_packet.arg1 == None
        assert scp_packet.arg2 == None
        assert scp_packet.arg3 == None
        assert scp_packet.data == \
            b'\xB7\xB7\xA5\xA5\xFE\xCA\xFE\xCA\x7B\x7B\x5A\x5A' + \
            b'\xFE\xED\xDE\xAF\x01'

        # Check that the bytestring this packet creates is the same as the one
        # we specified before.
        assert scp_packet.bytestring == packet

    def test_from_bytestring_1_args(self):
        """Test creating a new SCPPacket from a bytestring."""
        #     arg1: 0xA5A5B7B7
        #     data: 0xCAFECAFE5A5A7B7BFEEDDEAF01
        packet = b'\x87\xf0\xef\xee\xa5\x5a\x0f\xf0\xAD\xDE\xEF\xBE' + \
                 b'\xB7\xB7\xA5\xA5\xFE\xCA\xFE\xCA\x7B\x7B\x5A\x5A' + \
                 b'\xFE\xED\xDE\xAF\x01'
        sdp_packet = SDPPacket.from_bytestring(packet)
        scp_packet = SCPPacket.from_sdp_packet(sdp_packet, n_args=1)

        assert scp_packet.arg1 == 0xA5A5B7B7
        assert scp_packet.arg2 == None
        assert scp_packet.arg3 == None
        assert scp_packet.data == \
            b'\xFE\xCA\xFE\xCA\x7B\x7B\x5A\x5A' + \
            b'\xFE\xED\xDE\xAF\x01'

        # Check that the bytestring this packet creates is the same as the one
        # we specified before.
        assert scp_packet.bytestring == packet

    def test_from_bytestring_2_args(self):
        """Test creating a new SCPPacket from a bytestring."""
        #     arg1: 0xA5A5B7B7
        #     arg2: 0xCAFECAFE
        #     data: 0x5A5A7B7BFEEDDEAF01
        packet = b'\x87\xf0\xef\xee\xa5\x5a\x0f\xf0\xAD\xDE\xEF\xBE' + \
                 b'\xB7\xB7\xA5\xA5\xFE\xCA\xFE\xCA\x7B\x7B\x5A\x5A' + \
                 b'\xFE\xED\xDE\xAF\x01'
        sdp_packet = SDPPacket.from_bytestring(packet)
        scp_packet = SCPPacket.from_sdp_packet(sdp_packet, n_args=2)

        assert scp_packet.arg1 == 0xA5A5B7B7
        assert scp_packet.arg2 == 0xCAFECAFE
        assert scp_packet.arg3 == None
        assert scp_packet.data == b'\x7B\x7B\x5A\x5A\xFE\xED\xDE\xAF\x01'

        # Check that the bytestring this packet creates is the same as the one
        # we specified before.
        assert scp_packet.bytestring == packet

    def test_values(self):
        """Check that SCP packets respect data values."""
        with pytest.raises(ValueError):  # cmd_rc is 16 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1 << 16, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):  # cmd_rc is 16 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      -1, 0, 0, 0, 0, b'')

        with pytest.raises(ValueError):  # seq is 16 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 1 << 16, 0, 0, 0, b'')

        with pytest.raises(ValueError):  # seq is 16 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, -1, 0, 0, 0, b'')

        with pytest.raises(ValueError):  # arg1 is 32 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, 1 << 32, 0, 0, b'')

        with pytest.raises(ValueError):  # arg1 is 32 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, -1, 0, 0, b'')

        with pytest.raises(ValueError):  # arg2 is 32 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, 0xFFFFFFFF, 1 << 32, 0, b'')

        with pytest.raises(ValueError):  # arg2 is 32 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, 0xFFFFFFFF, -1, 0, b'')

        with pytest.raises(ValueError):  # arg3 is 32 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1 << 32, b'')

        with pytest.raises(ValueError):  # arg3 is 32 bits
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, -1, b'')

        with pytest.raises(ValueError):  # data is max 256 bytes
            SCPPacket(False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                      257*b'\x00')


class TestSDPFilter(object):
    """SDPFilters are used to determine whether a given packet should be passed
    to a callback or not."""
    def test_filter_x(self):
        """Test the trying to create a filter using an invalid slot fails."""
        with pytest.raises(ValueError):
            SDPFilter(x=3)

    def test_tag_filter(self):
        """Test filtering SDP packets based on IPTag."""
        # Create a new filters which accept varying values of IPTag
        no_filter = SDPFilter(tag=None)
        filter_0 = SDPFilter(tag=0)
        filter_1_to_3 = SDPFilter(tag=lambda x: 1 <= x <= 3)

        # Apply these filters to 4 different packets (with IPTags 0..3)
        sdp1 = SDPPacket(False, 0, 1, 0, 0, 0, 1, 1, 0, 0, b'')
        sdp2 = SDPPacket(False, 1, 1, 0, 0, 0, 1, 1, 0, 0, b'')
        sdp3 = SDPPacket(False, 2, 1, 0, 0, 0, 1, 1, 0, 0, b'')
        sdp4 = SDPPacket(False, 3, 1, 0, 0, 0, 1, 1, 0, 0, b'')

        assert no_filter(sdp1)
        assert no_filter(sdp2)
        assert no_filter(sdp3)
        assert no_filter(sdp4)

        assert filter_0(sdp1)
        assert not filter_0(sdp2)
        assert not filter_0(sdp3)
        assert not filter_0(sdp4)
        
        assert not filter_1_to_3(sdp1)
        assert filter_1_to_3(sdp2)
        assert filter_1_to_3(sdp3)
        assert filter_1_to_3(sdp4)

    def test_tag_src_x_y(self):
        """Test filtering SDP packets based on Src X and Y."""
        # Create a new filter with ranged X and Y.
        xy_filter = SDPFilter(src_x=lambda x: 0 <= x < 4,
                              src_y=lambda y: 0 <= y < 4)

        # Create SDP packets within this range, check that they all pass
        for x in range(4):
            for y in range(4):
                sdp = SDPPacket(False, 0, 0, 0, 0, 0, 0, 0, x, y, b'')
                assert xy_filter(sdp)

        # Check an SDP packet outside the range
        sdp = SDPPacket(False, 0, 0, 0, 0, 0, 0, 0, x + 1, y + 1, b'')
        assert not xy_filter(sdp)
