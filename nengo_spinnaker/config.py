import nengo
import nengo.config


@nengo.config.configures(nengo.Node)
class ConfigNode(nengo.config.ConfigItem):
    # TODO Raise Exceptions if the Node cannot possibly be f(t)
    f_of_t = nengo.config.Parameter(False)  # Node is a function of time only
    f_period = nengo.config.Parameter(None)  # Period, None means aperiodic


class Config(nengo.config.Config):
    config_items = [ConfigNode]  # add other ConfigItems here
