import pytest


@pytest.fixture(scope='session')
def spinnaker_ip(request):
    return request.config.getvalue('spinnaker')


def pytest_addoption(parser):
    # Add the option to run tests against a SpiNNaker machine
    parser.addoption("--spinnaker", help="Run tests on a SpiNNaker machine. "
                                         "Specify the IP address or hostname "
                                         "of the SpiNNaker machine to use.")


def pytest_runtest_setup(item):
    # Skip tests if a SpiNNaker board isn't specified
    if (getattr(item.obj, 'spinnaker', None) and
            item.config.getvalue('spinnaker') is None):
        pytest.skip("No SpiNNaker machine specified.")
