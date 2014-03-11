from pacman103.scp import sdp
import struct


class SDPMessageWithArgs(sdp.SDPMessage):
    def __init__(self, packed=None, **kwargs):
        # Construct the standard argument header
        self.cmdr = 0x00000000 if not "cmd_rc" in kwargs else kwargs["cmd_rc"]
        self.arg1 = 0x00000000 if not "arg1" in kwargs else kwargs["arg1"]
        self.arg2 = 0x00000000 if not "arg2" in kwargs else kwargs["arg2"]
        self.arg3 = 0x00000000 if not "arg3" in kwargs else kwargs["arg3"]

        # Initialise
        super(SDPMessageWithArgs, self).__init__(packed, **kwargs)

    def __str__(self):
        """
        Constructs a string that can be sent over a network socket using the
        member variables.  The command arguments are inserted at the start of
        the data.

        :returns: encoded string
        """
        return self._pack_hdr() + self._pack_args() + self.data

    def _pack_args(self):
        """
        Construct the argument and command code section of the data payload.
        """
        return struct.pack('<4I', self.cmdr, self.arg1, self.arg2, self.arg3)
