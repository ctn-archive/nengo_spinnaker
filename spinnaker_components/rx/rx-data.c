/*
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 * 
 * \addtogroup rxcomponent
 * @{
 */

#include "rx.h"

uint n_dimensions, dt, ticks_per_output, n_current_output, *keys;
value_t *values;

/**
 * \brief Copy in system region data.
 * 
 * Value (Type) | Description
 * ------------ | -----------
 * n_dimensions (```uint```) | Number of output dimensions <= 64
 * dt (```uint```) | Time period of simulation in microseconds
 * 
 */
void copy_in_system_region( address_t addr ) {
  n_dimensions = addr[0]; // Number of dimensions to represent
  dt = addr[1];           // Time step in us

  // Calculate the number of ticks between transmitting each output
  // packet.
  // Zero the index of the current output
  ticks_per_output = dt / n_dimensions;
  n_current_output = 0;

  // Allocate space for keys and values
  keys = spin1_malloc( sizeof( uint ) * n_dimensions );
  values = spin1_malloc( sizeof( value_t ) * n_dimensions );
}

/**
 * \brief Copy in routing keys data
 *
 * Complete routing keys (```uint```) to be used when transmitting 
 * dimensional data.  These are normally formed as:
 * 
 * ```( x << 24 ) | ( y << 16 ) | ( (p-1) << 11 ) | ( i << 6 ) | d```
 * 
 * Where ```x```, ```y``` and ```p``` refer to the processor where the Rx
 * component resides; ```i``` refers to the index of the connection/edge the data
 * is associated with (e.g., a given Rx component may feed multiple sinks with
 * different sets of dimensions) and ```d``` refers to the specific dimension
 * being transmitted.
 */
void copy_in_keys( address_t addr ) {
  spin1_memcpy( keys, addr, n_dimensions * sizeof( uint ) );
}

/**
 * \brief Copy in initial values data
 *
 * The initial values to transmit (as there may be some lag between the start of
 * the simulation and the first SDP packets arriving).  These must be in the same
 * order as the routing keys.
 */
void copy_in_initial_values( address_t addr ) {
  spin1_memcpy( values, addr, n_dimensions * sizeof( value_t ) );
}

/** @} */
