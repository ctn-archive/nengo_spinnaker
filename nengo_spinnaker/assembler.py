import logging

logger = logging.getLogger(__name__)


class Assembler(object):
    """An Assembler is the final stage in the build process prior to
    partitioning, placing and routing the model for simulation on SpiNNaker.

    The Assembler acts as an object builder which replaces instances of various
    objects in the connectivity tree that defines a simulation.  The Assembly
    process results in a connection tree which can be folded into a graph for
    partitioning, placing, routing and executing on a SpiNNaker machine.
    """
    object_assemblers = dict()

    @classmethod
    def add_object_assembler(cls, object_type, assembler):
        """Register a new object assembler.

        :param type object_type: The class of objects to assemble with this
                                 function.
        :param callable assembler: A callable which will be used to assemble
                                   objects of this type.  The callable should
                                   accept the `object`, `connection_trees`,
                                   `config`, `rngs`, `runtime`, `dt`, and
                                   `machine_timestep`.  Further documentation
                                   on these parameters is elsewhere in this
                                   class.
        """
        cls.object_assemblers[object_type] = assembler

    @classmethod
    def object_assembler(cls, object_type):
        """Decorator which marks a function as assembling a type of object.

        :param type object_type: The class of objects to assemble.
        """
        def dec(f):
            cls.add_object_assembler(object_type, f)
            return f
        return dec

    @classmethod
    def assemble_obj(cls, obj, connection_trees, config, rngs, runtime, dt,
                     machine_timestep):
        """Call the assembler for the given object.
        """
        # Work through the MRO to find an assembler function
        for c in obj.__class__.__mro__:
            if c in cls.object_assemblers:
                return cls.object_assemblers[c](
                    obj, connection_trees, config, rngs, runtime, dt,
                    machine_timestep)
        else:
            # Otherwise just return the object unchanged
            return obj

    @classmethod
    def assemble(cls, connection_trees, config, rngs, runtime, dt,
                 machine_timestep):
        """Assemble the objects in the given connectivity tree.

        :type connection_trees:
            :py:class:`~..connections.connection_tree.ConnectionTree`
        :return: Tuple of a list of objects and list of connections.
        :rtype: :func:`tuple`
        """
        # Get a list of all the objects we need to build from the original
        # connectivity tree, then progressively replace objects.
        logger.info("Assembly step 1/3: Assembling objects")
        connected_objects = connection_trees.get_objects()
        replaced_objects = dict()

        for obj in connected_objects:
            logger.debug("Assembling {}".format(obj))
            replaced_objects[obj] = cls.assemble_obj(
                obj, connection_trees, config, rngs, runtime, dt,
                machine_timestep)

        # Replace assembled objects in the connection tree
        logger.info("Assembly step 2/3: Adding assembled objects to "
                    "connectivity tree")
        connection_trees = connection_trees.get_new_tree_with_replaced_objects(
            replaced_objects)

        # Fold the connectivity tree by getting a list of objects and a list of
        # connections.
        logger.info("Assembly step 3/3: Folding connectivity tree, retrieving "
                    "objects")
        objs = connection_trees.get_objects()
        edges = connection_trees.get_folded_edges()

        # Return the objects and connections for the folded connectivity tree
        return (objs, edges)
