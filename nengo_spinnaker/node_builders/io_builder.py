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

    def __enter__(self):
        """Perform any tasks necessary to communicate with the live simulation
        """
        raise NotImplementedError

    def __exit__(self, exception_type, exception_val, trace):
        """Perform any tasks necessary to end communication with the simulation
        """
        raise NotImplementedError
