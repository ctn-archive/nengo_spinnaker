import ConfigParser
import sys

from pacman103.core import control

from . import builder


class Simulator(object):
    def __init__(self, model, machine_name=None, dt=0.001, seed=None):
        # Build the model
        self.builder = builder.Builder()
        self.dao = self.builder(model, dt, seed)
        self.dao.writeTextSpecs = True

        if machine_name is None:
            from pacman103 import conf
            try:
                machine_name = conf.config.get("Machine", "machineName")
            except ConfigParser.Error:
                machine_name = None

            if machine_name is None or machine_name == "None":
                raise Exception("You must specify a SpiNNaker machine as "
                                "either an option to the Simulator or in a "
                                "PACMAN103 configuration file.")

        self.machine_name = machine_name

    def run(self, time):
        """Run the model, currently ignores the time."""
        self.controller = control.Controller(sys.modules[__name__],
                                             self.machine_name)

        # Preparation functions
        # Consider moving this into PACMAN103
        for vertex in self.dao.vertices:
            if hasattr(vertex, 'prepare_vertex'):
                vertex.prepare_vertex()

        self.controller.dao = self.dao
        self.dao.set_hostname(self.machine_name)
        self.controller.map_model()
        self.controller.generate_output()
        self.controller.load_targets()
        self.controller.load_write_mem()
        self.controller.run(self.dao.app_id)
