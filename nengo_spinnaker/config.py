import nengo
import nengo.config

class Config(nengo.config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.configures(nengo.Node)
        self[nengo.Node].set_param('aplx', nengo.config.Parameter(None))
