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
  uint n_filters;
  uint n_filter_keys;
} region_system_t;

/**
* \brief Copy in data pertaining to the system region of the Ensemble.
*
* We expect there to be 7 ```uint``` size pieces of information within the
* system region (region 1). These are:
*
* Description | Units | Type | Becomes
* ----------- | ----- | ---- | -------
* Number of input dimensions | | ```uint``` | ::n_input_dimensions
* Number of output dimensions | | ```uint``` | ::n_output_dimensions
* Number of neurons | | ```uint``` | ::n_neurons
* Machine time step | Microseconds | ```uint``` | ::dt
* Refactory time constant | Steps | ```uint``` | ::t_ref
* dt over membrane time constant | | ```accum``` | ::dt_over_t_rc
* Number of filters | | ```uint``` |
* Number of filter keys | | ```uint``` |
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

bool data_get_filters(
  address_t addr,
  region_system_t *pars
);

bool data_get_filter_keys(
  address_t addr,
  region_system_t *pars
);

#endif

/** @} */
