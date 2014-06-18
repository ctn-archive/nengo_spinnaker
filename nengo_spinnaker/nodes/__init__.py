from ethernet import Ethernet

try:
    from serial import Serial, SpIOUART, NSTSpiNNlink
except ImportError:
    Serial = None
    SpIOUART = None
    NSTSpiNNlink = None
