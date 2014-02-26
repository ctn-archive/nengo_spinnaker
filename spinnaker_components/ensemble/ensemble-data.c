/*****************************************************************************

SpiNNaker and Nengo Integration

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

#include "spin-nengo-ensemble.h"

bool copy_in_system_region( address_t addr ){
  /* We have 7 parameters of a word each in the order:
   * 1. Number of input dimensions D_in
   * 2. Number of output dimensions D_out
   * 3. Number of neurons N
   * 4. dt in useconds
   * 5. tau_ref
   * 6. tau_rc
   * 7. Filter decay constant
   */
  n_input_dimensions  = addr[0];
  n_output_dimensions = addr[1];
  n_neurons           = addr[2];
  dt                  = addr[3];
  t_ref               = addr[4];
  one_over_t_rc       = addr[5];
  filter              = addr[6];

  return true;
}

bool copy_in_bias( address_t addr ){
  /* Biases are 1xN, thus we just copy this area of memory. */
  spin1_memcpy( i_bias, addr, n_neurons * sizeof( accum ) );
  return true;
}

bool copy_in_encoders( address_t addr ){
  /* Encoders are a N x D_in matrix. */
  spin1_memcpy( encoders, addr, n_neurons * n_input_dimensions * sizeof( accum ) );
  return true;
}

bool copy_in_decoders( address_t addr ){
  /* Decoders are a N x D_out matrix. */
  spin1_memcpy( decoders, addr, n_neurons * n_output_dimensions * sizeof( accum ) );
  return true;
}

bool copy_in_decoder_keys( address_t addr ){
  /* Biases are 1xN. */
  spin1_memcpy( output_keys, addr, n_output_dimensions * sizeof( uint ) );
  return true;
}
