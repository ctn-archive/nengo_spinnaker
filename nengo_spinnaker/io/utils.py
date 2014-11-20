"""Utilities for IO builders and handlers.
"""


def stop_on_keyboard_interrupt(f):
    """A decorator that (when used within a class) will call the `stop` method
    of the class if a `KeyboardInterrupt` is raised during execution of the
    function.

    :param function f: Function to wrap.
    """
    def f_(self, *args):
        try:
            # Call the original function with the given arguments.
            f(self, *args)
        except KeyboardInterrupt:
            # Call the stop method
            self.stop()

    return f_
