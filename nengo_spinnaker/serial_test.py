import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=8000000, rtscts=True)
ser.write("S+\n")
print ser
while True:
    print ser.readline(),
    #print repr(ser.read())
