from ethernet import Ethernet

try:
    from uart import UART, SpIOUARTProtocol, NSTSpiNNlinkProtocol
except ImportError:
    UART = None
    SpIOUARTProtocol = None
    NSTSpiNNlinkProtocol = None
