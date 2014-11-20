import nengo
import nengo.config
import nengo.params


class FunctionOfTimeBoolParam(nengo.params.BoolParam):
    def __init__(self):
        super(FunctionOfTimeBoolParam, self).__init__(False)

    def validate(self, instance, boolean):
        super(FunctionOfTimeBoolParam, self).validate(instance, boolean)
        if instance._configures.size_in > 0:
            raise AttributeError('Cannot create a function-of-time node for a '
                                 'Node which is not a function of time.')


class Config(nengo.config.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.configures(nengo.Node)
        self[nengo.Node].set_param('f_of_t', FunctionOfTimeBoolParam())
        self[nengo.Node].set_param(
            'f_period', nengo.params.NumberParam(None, optional=True))
