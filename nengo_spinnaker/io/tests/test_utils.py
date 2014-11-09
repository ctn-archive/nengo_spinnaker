import pytest

from .. import utils as io_utils


def test_stop_on_keyboard_interrupt():
    """Check that we can decorate functions so that a keyboard interrupt is
    caught and calls a given method.
    """
    # Create an object we can test with
    class TestObject(object):
        stop_fn_called = False

        @io_utils.stop_on_keyboard_interrupt
        def test(self, a, b, c):
            """A function that we want to call periodically."""
            # Assert the inputs were correct
            assert a == 1
            assert b == 2
            assert c == 3

            # Raise a keyboard interrupt to test the stop method
            raise KeyboardInterrupt

        def stop(self):
            self.stop_fn_called = True

    test_obj = TestObject()

    # Now call the test function with an incorrect value to check that we get
    # an assertion error passed through.
    with pytest.raises(AssertionError):
        test_obj.test(0, 1, 2)
    assert not test_obj.stop_fn_called

    # Call the test function with the correct arguments to check that the
    # keyboard interrupt is handled correctly.
    try:
        test_obj.test(1, 2, 3)
    except KeyboardInterrupt:
        assert False, "KeyboardInterrupt was not caught."
    assert test_obj.stop_fn_called
