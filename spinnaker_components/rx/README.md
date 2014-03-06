SpiNNaker / Nengo Integration
=============================

Rx Component
------------

The Rx component exists to allow the **host machine** to inject
multi-dimensional values into a running SpiNNaker simulation.  Each Rx
component is capable of injecting 64 dimensions-worth of data into a running
simulation.

## Operation

An Rx component stores:

1. An array of routing keys
2. An array of cached values, ```D_out```

If the number of output dimensions is ```|D_out|```, then one output dimension
value is transmitted each ```dt / |D_out|``` seconds. The current output
dimension ```i``` is used to index keys and values.

On receipt of an SDP packet with ```cmd_rc = 0x00000001``` the current stored
values are replaced by those in the data payload of the packet. Since a packet
may have up to 256 bytes of payload, this allows of 64 dimensions.

## Regions and Data

### 1. System Region

Value (Type) | Description
------------ | -----------
n_dimensions (```uint```) | Number of output dimensions <= 64
dt (```uint```) | Time period of simulation in microseconds

### 2. Routing Keys

Complete routing keys (```uint```) to be used when transmitting 
dimensional data.  These are normally formed as:

```( x << 24 ) | ( y << 16 ) | ( (p-1) << 11 ) | ( i << 6 ) | d```

Where ```x```, ```y``` and ```p``` refer to the processor where the Rx
component resides; ```i``` refers to the index of the connection/edge the data
is associated with (e.g., a given Rx component may feed multiple sinks with
different sets of dimensions) and ```d``` refers to the specific dimension
being transmitted.

### 3. Initial Values

The initial values to transmit (as there may be some lag between the start of
the simulation and the first SDP packets arriving).  These must be in the same
order as the routing keys.

## SDP Packet Format

1. ```cmd_rc``` must be ```0x00000001```
2. ```data``` must be an array of appropriate values to be transmitted, in the
   same order as the routing keys with which they must be associated.
