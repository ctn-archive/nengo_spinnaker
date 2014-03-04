import socket
import struct

port = 17899

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', port))
while True:
    data, addr = s.recvfrom(1024)
    print '%s: %s'%(addr, `data`)
    arg1, arg2 = struct.unpack('<II', data[14:22])
    print '%s: 0x%08x 0x%08x'%(addr, arg1, arg2)


