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

#ifndef __DIMENSIONAL_IO_H__
#define __DIMENSIONAL_IO_H__

#include "common-typedefs.h"
#include "spin1_api.h"

typedef struct filtered_input_buffer {
  //! Represents a filtered input buffer
  uint d_in;            //!< Number of input dimensions, D_in

  accum *accumulator; //!< Accumulates input values, a 1xD_in matrix
  accum *filtered;    //!< Holds the filtered value, a 1xD_in matrix

  accum filter;       //!< Filter value, e.g., \f$\exp(-\frac{dt}{\tau})\f$
  accum n_filter;     //!< 1 - filter value
} filtered_input_buffer_t;

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

  return buffer;
}

//! Filter an input buffer and zero the accumulator
static inline void input_buffer_step( filtered_input_buffer_t *buffer ) {
  // For each input dimension perform filtering and zero the accumulator
  for( uint d = 0; d < buffer->d_in; d++ ){
    // Perform the filtering
    buffer->filtered[d] *= buffer->filter;
    buffer->filtered[d] += buffer->accumulator[d] * buffer->n_filter;

    // Zero the accumulator
    buffer->accumulator[d] = 0;
  }
}

#endif
