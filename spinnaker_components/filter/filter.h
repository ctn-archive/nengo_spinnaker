/**
 * \addtogroup Filter
 * \brief A component which filters its input and provides output of that
 *        filtered value at regular intervals.
 *
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 * @{
 */

#ifndef __FILTER_H__
#define __FILTER_H__

#include "spin1_api.h"
#include "filtered-input.h"
#include "nengo-common.h"

#include "common-impl.h"

/** \brief Shared filter parameters.
  */
typedef struct filter_parameters {
  uint machine_timestep;   //!< Machine time step / useconds
  uint transmission_delay; //!< Number of ticks between output transmissions

  uint interpacket_pause;  //!< Delay in usecs between transmitting packets

  uint n_dimensions;       //!< Number of dimensions to represent

  value_t *input;          //!< Input buffer
  uint *keys;              //!< Output keys
} filter_parameters_t;
extern filter_parameters_t g_filter; //!< Global parameters

bool data_system(address_t addr);
bool data_get_output_keys(address_t addr);

#endif

/** @} */
