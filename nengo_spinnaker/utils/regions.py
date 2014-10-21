"""Nengo Specific Region Types
"""
import math

from ..spinnaker import regions


class BitfieldBasedRecordingRegion(regions.Region):
    def __init__(self, n_ticks):
        super(BitfieldBasedRecordingRegion, self).__init__(
            in_dtcm=False, unfilled=True)
        self.n_ticks = n_ticks if n_ticks is not None else 0

    def sizeof(self, vertex_slice):
        # Size is the number of words required x number of ticks
        frame_length = int(math.ceil(
            (vertex_slice.stop - vertex_slice.start) / 32.))
        return frame_length * self.n_ticks
