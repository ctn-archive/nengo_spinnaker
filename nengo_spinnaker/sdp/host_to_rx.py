from . import sdp_message
from pacman103.lib import parameters


class HostToRxPacket(sdp_message.SDPMessageWithArgs):
    """
    An SDP packet used to update the value transmitted into the simulation by
    an Rx element.

    :param edge: The Edge on which this data is to be transmitted.
    :param data: The n-dimensional value to be injected by the Rx component.
    """
    def __init__(self, edge, data=[]):
        if len(data) > 64:
            raise ValueError(
                "An Rx component cannot represent more than 64 dimensions.\n"
            )

        # TODO: Insert reasons as to why this is valid...
        # Get the set of subvertices from which this edge originates, it
        # should consist of 1 element.  This is the subvertex to which we wish
        # to send the given data.
        subvertices = set(map(lambda se: se.presubvertex, edge.subedges))
        assert(len(subvertices) == 1)
        subvertex = subvertices[0]

        # Format the arguments and data
        assert(edge.width == len(data))
        arg1 = edge.start
        arg2 = edge.width
        fixed_data = [v.converted for v in parameters.S1615(data)]
        data = ''.join(fixed_data)

        # Get the co-ordinates of the Rx vertex we're communicating with
        x, y, p = subvertex.placement.processor.get_coordinates()

        # Initialise
        super(HostToRxPacket, self).__init__(
            dst_x=x, dst_y=y, dst_cpu=p,                 # Packet destination
            cmd_rc=0x1, arg1=arg1, arg2=arg2, data=data  # Packet data
        )
