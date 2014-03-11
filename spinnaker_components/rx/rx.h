/**
 * \addtogroup rxcomponent SpiNNaker Rx Component
 * \brief A component which receives SDP packets from the host and translates
 *        them into MC packets to stimulate other Nengo components.
 *
 * The Rx component exists to allow the **host machine** to inject
 * multi-dimensional values into a running SpiNNaker simulation.  Each Rx
 * component is capable of injecting 64 dimensions-worth of data into a running
 * simulation.
 * 
 * Operation
 * ---------
 * 
 * An Rx component stores:
 * 
 * 1. An array of routing keys (see ::copy_in_keys)
 * 2. An array of cached values, ```D_out``` (see ::copy_in_initial_values)
 * 
 * If the number of output dimensions is ```|D_out|```, then one output dimension
 * value is transmitted each ```dt / |D_out|``` seconds. The current output
 * dimension ```i``` is used to index keys and values.
 * 
 * On receipt of an SDP packet with ```cmd_rc = 0x00000001``` the current stored
 * values are replaced by those in the data payload of the packet. Since a packet
 * may have up to 256 bytes of payload, this allows of 64 dimensions.
 * 
 * ## SDP Packet Format
 * 
 * 1. ```cmd_rc``` must be ```0x00000001```
 * 2. ```data``` must be an array of appropriate values to be transmitted, in the
 *    same order as the routing keys with which they must be associated.
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 * \author Terry Stewart
 * 
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 */

#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "common-impl.h"
#include "common-typedefs.h"

// Typedefs
typedef accum value_t;

// Function prototypes
int c_main( void );

void copy_in_system_region( address_t addr );   // System region
void copy_in_keys( address_t addr );            // Keys
void copy_in_initial_values( address_t addr );  // Copy initial values

void timer_callback( uint simulation_time, uint none ); // Timer tick
void sdp_packet_received( uint mailbox, uint port );    // Handle SDP Rx

// System variables
/** @{ */
extern uint n_dimensions; //!< Number of dimensions associated with this Rx

extern uint ticks_per_output; //!< Number of ticks to wait between each output
extern uint n_current_output; //!< Index of current output

extern uint *keys;        //!< Keys to associate with outgoing values
extern value_t *values;   //!< Most recently cached output value

/** @} */
