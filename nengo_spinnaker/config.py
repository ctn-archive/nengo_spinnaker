import nengo
import nengo.config

@nengo.config.configures(nengo.Node)
class ConfigNode(nengo.config.ConfigItem):
    aplx = nengo.config.Parameter(None)
    # add more parameters here as needed

# create other ConfigItem classes for configuring Ensembles
# or Connections (or Models or whatever)

class Config(nengo.config.Config):
    config_items = [ConfigNode]  # add other ConfigItems here

