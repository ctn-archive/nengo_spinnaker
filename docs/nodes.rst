===================================
Running Nodes directly on SpiNNaker
===================================

By default Nodes are executed on the host computer and communicate with the
SpiNNaker board to transmit and receive values.  The result can be undesirable
sampling of Node input and output.

For example::

    import nengo
    import nengo_spinnaker

    model = nengo.Network()
    with model:
        n = nengo.Node(np.sin)
        e = nengo.Ensemble(nengo.LIF(100), 1)
        p = nengo.Probe(e)

        nengo.Connection(n, e)

    sim = nengo_spinnaker.Simulator(model)
    sim.run(10.)

Results in:

.. image:: http://amundy.co.uk/assets/img/nengo_spinnaker/host-node.png
    :width: 500px

For Nodes which are solely functions of time it is possible to precompute the
output of the Node and play this back.  Nodes with a constant output value and
no input are automatically added to the bias current of Ensemble which they
feed.  Finally, more complex Nodes may be implemented as SpiNNaker executables
and directly executed on the SpiNNaker hardware.

Function-of-time Nodes
======================

Nodes which are purely functions of time may be precomputed for the duration of
the simulation (or the period of the function if appropriate) and played back
during the simulation.  Nodes you wish to be executed in this way must be
marked with an appropriate directive::

    # As before...

    # Create the configuration and configure `n` as being f(t)
    config = nengo_spinnaker.Config()
    config[n].f_of_t = True  # Mark Node as being a function of time
    config[n].f_period = 2*np.pi  # Mark the period of the function

    # Pass the configuration to the simulator
    sim = nengo_spinnaker.Simulator(model, config=config)

Results in:

.. image:: http://amundy.co.uk/assets/img/nengo_spinnaker/spinn-node.png
    :width: 500px


The two directives are:
 * `f_of_t` marks a Node as being precomputable.  This is not checked - be
   careful!
 * `f_period` marks the period of the function in seconds.  If this is `None`
   then the Node will be precomputed for the entire duration of the simulation
   - it is possible to run out of memory.  Again, this cannot be trivially
   validated.


Writing a SpiNNaker executable for a Node
=========================================

Using the binary
================

Configuring the executable instances
====================================


=================================
Writing new Input/Output Handlers
=================================

Input/Output Handlers manage the communication between the host and the
SpiNNaker machine running the simulation.  This entails two tasks:

1. Modifying the SpiNNaker model to include appropriate executables and
   connections for handling Node input/output.
2. Providing functions for getting input for Nodes and setting Node output.

The first of these tasks is handled by "Node Builders", the second by "Node
Communicators".

Node Builders
=============

When building a model for simulation a :py:class:`nengo_spinnaker.builder.Builder`
delegates the tasks of building Nodes and the connections into or out of Nodes
to a Node Builder.

Additionally, the :py:class:`nengo_spinnaker.Simulator` will expect the Node
Builder to provide a context manager for the Node Communicator.

A Node Builder is expected to look like the following:

.. py:class:: GenericNodeBuilder

    .. py:method:: get_node_in_vertex(self, builder, connection)

        Get the PACMAN vertex where input to the Node should be sent.

        :param builder: A :py:class:`nengo_spinnaker.builder.Builder` instance providing
            `add_vertex` and `add_edge` methods.
        :param connection: A :py:class:`nengo.Connection` object which specifies the connection
            being built.  The Node will be referred to by `connection.post`.
        :returns: The PACMAN vertex where input for the Node at the end of the
            given connection should be sent.


        It is expected that this function will need to create new PACMAN
        vertices and add them to the graph using the builder object.

    .. py:method:: get_node_out_vertex(self, builder, connection)

        Get the PACMAN vertex where output from the Node can be expected to
        arrive in the SpiNNaker network.

        :param builder: A :py:class:`nengo_spinnaker.builder.Builder` instance providing
            `add_vertex` and `add_edge` methods.
        :param connection: A :py:class:`nengo.Connection` object which specifies the connection
            being built.  The Node will be referred to by `connection.pre`.
        :returns: The PACMAN vertex where output from the Node will appear.

        It is expected that this function will need to create new PACMAN
        vertices and add them to the graph using the builder object.

    .. py:method:: build_node(self, builder, node)

        Perform any tasks necessary to build a Node which is neither constant
        nor a function of time.

        :param builder: A :py:class:`nengo_spinnaker.builder.Builder` instance providing
            `add_vertex` and `add_edge` methods.
        :param node: The :py:class:`nengo.Node` object for which to provide IO.

        .. note::
            In all current implementations this method does nothing, it is
            generally more useful to instantiate any vertices or edges when
            connecting to or from a Node.

    .. py:attribute:: io

        A reference to the Communicator object.

    .. py:method:: __enter__(self)

        Create and return a Communicator to handle input/output for Nodes.

        :returns: A Communicator of the appropriate type.

    .. py:method:: __exit__(self, exception_type, exception_value, traceback)

        Perform any tasks necessary to stop the Communicator from running.


Node Communicators
==================

The :py:class:`nengo_spinnaker.Simulator` delegates the task of getting Node
input and setting Node output to a communicator which is generated by the Node
Builder.

A Node Communicator is required to look like the following:

.. py:class:: GenericNodeCommunicator

    .. warning::
        It is required that the Communicator be thread safe.  Each Node is
        independently responsible for getting its input and setting its output
        and each Node will be executed within its own thread.

    .. py:method:: start(self)

        Start execution of the communicator thread.

    .. py:method:: get_node_input(self, node)

        Return the latest received input for the given Node.

        :param node: A :py:class:`nengo.Node` for which input is desired.
        :returns: The latest received value as a Numpy array, or
            `None` if no data has yet been received from the Node.
        :raises: :py:exc:`KeyError` if the Node is not recognised by the
            Communicator.

    .. py:method:: set_node_output(self, node, output)

        Transmit the output of the Node to the SpiNNaker board.

        :param node: A :py:class:`nengo.Node` for which output is being
            provided.
        :param output: The latest output from the Node.
        :raises: :py:exc:`KeyError` if the Node is not recognised by the
            Communicator.
