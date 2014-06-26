/**
 * Ensemble - Data
 *
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
 * \addtogroup ensemble
 * @{
 */

#include "ensemble.h"
#include "common-impl.h"

#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__

/** \brief Representation of system region. See ::data_system. */
typedef struct region_system {
  uint n_input_dimensions;
  uint n_output_dimensions;
  uint n_neurons;
  uint machine_timestep;
  uint t_ref;
  value_t dt_over_t_rc;
  bool record_spikes;
  uint n_inhibitory_dimensions;
} region_system_t;

/**
* \brief Copy in data pertaining to the system region of the Ensemble.
*/
bool data_system( address_t addr );

bool data_get_bias(
  address_t addr,
  uint n_neurons
);

bool data_get_encoders(
  address_t addr,
  uint n_neurons,
  uint n_input_dimensions
);

bool data_get_decoders(
  address_t addr,
  uint n_neurons,
  uint n_output_dimensions
);

bool data_get_keys(
  address_t addr,
  uint n_output_dimensions
);

#endif

/** @} */
