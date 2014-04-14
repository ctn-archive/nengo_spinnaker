import sys

from pacman103.core import control
from pacman103 import conf

from . import builder


class Simulator(object):
    def __init__(self, model, dt=0.001, seed=None, use_serial=False):
        # Build the model
        self.builder = builder.Builder()
        self.dao = self.builder(model, dt, seed, use_serial=use_serial)
        self.dao.writeTextSpecs = True

    def run(self, time):
        """Run the model, currently ignores the time."""
        self.controller = control.Controller(
            sys.modules[__name__],
            conf.config.get('Machine', 'machineName')
        )

        # Preparation functions
        # Consider moving this into PACMAN103
        for vertex in self.dao.vertices:
            if hasattr(vertex, 'prepare_vertex'):
                vertex.prepare_vertex()

        self.controller.dao = self.dao
        self.dao.set_hostname(conf.config.get('Machine', 'machineName'))
        self.controller.map_model()
        if self.builder.use_serial:
            self.generate_serial_key_maps()
        self.controller.generate_output()
        self.controller.load_targets()
        self.controller.load_write_mem()
        self.controller.run(self.dao.app_id)

        if self.builder.use_serial:
            self.start_serial_io(time)

    def generate_serial_key_maps(self):
        self.serial_rx = {}
        self.serial_tx = {}
        for edge in self.builder.serial.in_edges:
            for subedge in edge.subedges:
                key = edge.prevertex.generate_routing_info(subedge)[0]
                node = edge.post
                self.serial_tx[key] = node

        for edge in self.builder.serial.out_edges:
            for subedge in edge.subedges:
                key = edge.prevertex.generate_routing_info(subedge)[0]
                node = edge.pre
                if node not in self.serial_rx:
                    self.serial_rx[node] = [key]
                else:
                    self.serial_rx[node].append(key)


    def start_serial_io(self, time):
        import serial
        import time

        self.serial = serial.Serial('/dev/ttyUSB0', baudrate=8000000,
                                    rtscts=True)
        self.serial.write("S+\n")   # send spinnaker packets to host
        start = time.time()
        buffer = {}
        while True:
            line = self.serial.readline().strip()
            print line
            if '.' not in line:
                continue
            parts = line.split('.')
            parts = [int(p,16) for p in parts]
            if len(parts) == 3:
                header, key, payload = parts

                if payload & 0x80000000:
                    payload -= 0x100000000
                value = (payload * 1.0) / (2**15)

                base_key = key & 0xFFFFF800
                d = (key & 0x0000003F)

                if base_key in self.serial_tx:
                    node = self.serial_tx[base_key]

                    vector = buffer.get(base_key, None)
                    if vector is None:
                        vector = [None]*node.size_in
                        buffer[base_key] = vector
                    vector[d] = value
                    if None not in vector:
                        now = time.time() - start
                        node.output(now, vector)
                        del buffer[base_key]

