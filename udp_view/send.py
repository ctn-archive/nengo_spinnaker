import socket

port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto('test', ('127.0.0.1', port))
