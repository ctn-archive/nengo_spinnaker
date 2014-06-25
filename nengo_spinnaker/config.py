import nengo
import nengo.config


class Config(nengo.config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.configures(nengo.Node)
        self[nengo.Node].set_param('f_of_t', nengo.config.Parameter(False))
        self[nengo.Node].set_param('f_period', nengo.config.Parameter(None))
