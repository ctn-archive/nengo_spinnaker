import nengo
import utils

from connection import IntermediateConnection


def reroute_modulatory_connections(objs, connections, probes):
    new_objs = list(objs)
    new_connections = list()

    # Loop through connections and their associated learning rules
    replaced_connections = list()
    for c in connections:
        intermediate_c = None
        replaced_learning_rules = list()

        for l in utils.connections.get_learning_rules(c):
            # If learning rule is PES
            if isinstance(l, nengo.PES):
                # Create an intermediate connection
                # To replace error connection
                e = IntermediateConnection.from_connection(l.error_connection)

                # Reroute this so it terminates at connection's pre-object
                e.post_obj = c.pre_obj

                # Add original error connection to list of
                # Connections that have been replaced
                replaced_connections.append(l.error_connection)

                # Add error connection to output
                new_connections.append(e)

                # If intermediate version of connection hasn't been created
                # Create one from c and wipe its exiting list of learning rules
                if intermediate_c is None:
                    intermediate_c = IntermediateConnection.from_connection(c)
                    intermediate_c.learning_rule = list()

                # Create new learning rule using intermediate error connection
                intermediate_c.learning_rule.append(
                    nengo.PES(e, l.learning_rate))

                # Add original learning rule to list list
                # Of learning rules that have been replaced
                replaced_learning_rules.append(l)

        # If this connection's been replaced
        if intermediate_c is not None:
            # Add learning rules from original connection that
            # Haven't been replaced to intermediate connection
            intermediate_c.learning_rule.extend(
                [l for l in utils.connections.get_learning_rules(c)
                    if l not in replaced_learning_rules])

            # Add original to list
            replaced_connections.append(c)

            # Add intermediate connection to output
            new_connections.append(intermediate_c)

    # Add connections from original list that
    # Haven't been replaced to output list
    new_connections.extend(
        [c for c in connections if c not in replaced_connections])

    # Return new lists
    return new_objs, new_connections
