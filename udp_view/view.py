import socket

port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('127.0.0.1', port))
while True:
    data, addr = s.recvfrom(1024)
    print '%s: %s'%(addr, `data`)


