/*****************************************************************************

SpiNNaker and Nengo Integration
-------------------------------

Tools for handling the receipt of multidimensional input values.

******************************************************************************

Authors:
 Andrew Mundy <mundya@cs.man.ac.uk> -- University of Manchester
 Terry Stewart			    -- University of Waterloo

Date:
 17-22 February 2014

******************************************************************************

Advanced Processors Technologies,   Computational Neuroscience Research Group,
School of Computer Science,         Centre for Theoretical Neuroscience,
University of Manchester,           University of Waterloo,
Oxford Road,                        200 University Avenue West,
Manchester, M13 9PL,                Waterloo, ON, N2L 3G1,
United Kingdom                      Canada

*****************************************************************************/

#include "dimensional-io.h"

//! Create and initialise an input buffer, and zero the accumulator
static inline filtered_input_buffer_t* input_buffer_initialise( uint d_in ) {
  // Create the buffer on the heap
  filtered_input_buffer_t *buffer = spin1_malloc(
    sizeof( filtered_input_buffer_t )
  );
  buffer->d_in = d_in;

  // Initialise the buffer accumulator and values
  buffer->accumulator = spin1_malloc( sizeof( accum ) * d_in );
  buffer->filtered = spin1_malloc( sizeof( accum ) * d_in );
}

//! Filter an input buffer and zero the accumulator
static inline void input_buffer_step( filtered_input_buffer_t *buffer ) {
  for( uint d = 0; d < buffer->d_in; d++ ){
    // Perform the filtering
    buffer->filtered[d] *= buffer->filter;
    buffer->filtered[d] += buffer->accumulator[d] * buffer->n_filter;

    // Zero the accumulator
    buffer->accumulator[d] = 0;
  }
}
