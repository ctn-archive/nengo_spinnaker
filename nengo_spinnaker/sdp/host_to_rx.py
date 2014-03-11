from . import sdp_message
from pacman103.lib import parameters


class HostToRxPacket(sdp_message.SDPMessageWithArgs):
    """
    An SDP packet used to update the value transmitted into the simulation by
    an Rx element.

    :param rx_vertex: The ReceiverVertex this packet is to be send to.
    :param data: The n-dimensional value to be injected by the Rx component.
    :param start: The offset into the vector where this data is to be added.

    ..todo::
        Change this so that we accept an Edge instead of a Vertex and hence
        determine the originating Subvertex for all Subedges of the Edge.
        This Subvertex is where we wish to transmit this packet.
    """
    def __init__(self, rx_vertex, data=[], start=0):
        if len(data) > 64:
            raise ValueError(
                "An Rx component cannot represent more than 64 dimensions.\n"
            )

        # Format the arguments and data
        arg1 = start
        arg2 = len(data)
        fixed_data = [v.converted for v in parameters.S1615(data)]
        data = ''.join(fixed_data)

        # Get the co-ordinates of the Rx vertex we're communicating with
        # For now assume that 1 Rx vertex maps to 1 subvertex...
        # TODO: * Correct this assumption
        #       * Talk to the PACMAN guys about making the 2nd line shorter!
        assert(len(rx_vertex.suvertices == 1))
        x, y, p = rx_vertex.subvertices.placement.processor.get_coordinates()

        # Initialise
        super(HostToRxPacket, self).__init__(
            dst_x=x, dst_y=y, dst_cpu=p,                 # Packet destination
            cmd_rc=0x1, arg1=arg1, arg2=arg2, data=data  # Packet data
        )
