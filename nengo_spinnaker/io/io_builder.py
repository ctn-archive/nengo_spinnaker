class IOBuilder(object):
    def __call__(self, builder):
        self.builder = builder
        return self

    def get_node_in_vertex(self, c):
        """Get the Vertex for input to the terminating Node of the given
        Connection
        """
        raise NotImplementedError

    def get_node_out_vertex(self, c):
        """Get the Vertex for output from the originating Node of the given
        Connection
        """
        raise NotImplementedError

    def get_node_input(self, node):
        """Return the latest input for the given Node

        :return: an array of data for the Node, or None if no data received
        :raises KeyError: if the Node is not a valid Node
        """
        raise NotImplementedError

    def set_node_output(self, node, output):
        """Set the output of the given Node

        Transmits the value given to the simulation
        :raises KeyError: if the Node is not a valid Node
        """
        raise NotImplementedError

    def __enter__(self):
        """Perform any tasks necessary to communicate with the live simulation
        """
        raise NotImplementedError

    def __exit__(self, exception_type, exception_val, trace):
        """Perform any tasks necessary to end communication with the simulation
        """
        raise NotImplementedError
