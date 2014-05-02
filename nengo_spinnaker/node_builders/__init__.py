from ethernet import Ethernet

try:
    from spinnlink import SpiNNlinkUSB
except ImportError:
    SpiNNlinkUSB = None
