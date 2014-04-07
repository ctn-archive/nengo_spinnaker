import sys

from pacman103.core import control
from pacman103 import conf

from . import builder


class Simulator(object):
    def __init__(self, model, dt=0.001, seed=None):
        # Build the model
        self.builder = builder.Builder()
        self.dao = self.builder(model, dt, seed)
        self.dao.writeTextSpecs = True

    def run(self, time):
        """Run the model, currently ignores the time."""
        self.controller = control.Controller(
            sys.modules[__name__],
            conf.config.get('Machine', 'machineName')
        )
        self.controller.dao = self.dao
        self.dao.set_hostname(conf.config.get('Machine', 'machineName'))
        self.controller.set_application_mask(0xFFFFFFE0)
        self.controller.map_model()
        self.controller.generate_output()
        self.controller.load_targets()
        self.controller.load_write_mem()
        self.controller.run(self.dao.app_id)
