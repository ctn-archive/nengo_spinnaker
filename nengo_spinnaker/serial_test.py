import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=8000000, rtscts=True)
ser.write("S+\n")
ser.write("E+\n")
ser.write("Z+\n")

#ser.write("000007C1.00000001\n");

print ser
while True:
    #print ser.readline(),
    print repr(ser.read())
