/*****************************************************************************

SpiNNaker and Nengo Integration

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 17-22 February, 3 March 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

#include "rx.h"

uint n_dimensions, dt, ticks_per_output, n_current_output, *keys;
value_t *values;

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

void copy_in_keys( address_t addr ) {
  spin1_memcpy( keys, addr, n_dimensions * sizeof( uint ) );
}

void copy_in_initial_values( address_t addr ) {
  spin1_memcpy( values, addr, n_dimensions * sizeof( value_t ) );
}
