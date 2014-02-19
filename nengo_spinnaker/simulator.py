from . import builder

class Simulator:
    def __init__(self, model, dt=0.001, seed=None):
        self.builder = builder.Builder(model, dt=dt, seed=seed)
        