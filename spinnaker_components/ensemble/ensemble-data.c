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
 */

#include "spin-nengo-ensemble.h"

/**
 * \brief Copy in data pertaining to the system region of the Ensemble.
 *
 * We expect there to be 7 ```uint``` size pieces of information within the
 * system region (region 1).  These are:
 *
 * Description | Units | Type | Becomes
 * ----------- | ----- | ---- | -------
 * Number of input dimensions | | ```uint``` | ::n_input_dimensions
 * Number of output dimensions | | ```uint``` | ::n_output_dimensions
 * Number of neurons | | ```uint``` | ::n_neurons
 * dt | Microseconds | ```uint``` | ::dt
 * Refactory time constant | Steps of dt | ```uint``` | ::t_ref
 * Inverse of membrane time constant | | ```accum``` | ::one_over_t_rc
 * Filter decay constant | | ```accum``` | ::filter
 */
bool copy_in_system_region( address_t addr ){
  n_input_dimensions  = addr[0];
  n_output_dimensions = addr[1];
  n_neurons           = addr[2];
  dt                  = addr[3];
  t_ref               = addr[4] << 28;
  one_over_t_rc       = kbits( addr[5] );
  filter              = kbits( addr[6] );

  return true;
}

/**
 * \brief Copy in bias currents.
 */
bool copy_in_bias( address_t addr ){
  /* Biases are 1xN, thus we just copy this area of memory. */
  spin1_memcpy( i_bias, addr, n_neurons * sizeof( accum ) );
  return true;
}

/**
 * \brief Copy in encoders
 *
 * We expect these to be in a \f$N \times D_{in}\f$ matrix.
 */
bool copy_in_encoders( address_t addr ){
  /* Encoders are a N x D_in matrix. */
  spin1_memcpy( encoders, addr, n_neurons * n_input_dimensions * sizeof( accum ) );
  return true;
}

/**
 * \brief Copy in decoders
 *
 * We expect these to be in a \f$N \times D_{out}\f$ matrix.
 */
bool copy_in_decoders( address_t addr ){
  /* Decoders are a N x D_out matrix. */
  spin1_memcpy( decoders, addr, n_neurons * n_output_dimensions * sizeof( accum ) );
  return true;
}

/**
 * \brief Copy in decoder keys
 *
 * These are used to associate each output value with an appropriate routing key.
 */
bool copy_in_decoder_keys( address_t addr ){
  /* Biases are 1xN. */
  spin1_memcpy( output_keys, addr, n_output_dimensions * sizeof( uint ) );
  return true;
}
