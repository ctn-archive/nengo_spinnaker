/**
 * Ensemble - Input
 * ----------------
 * Structures and functions to deal with arriving multicast packets (input).
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

#ifndef __ENSEMBLE_INPUT_H__
#define __ENSEMBLE_INPUT_H__

/* Structs *******************************************************************/
/**
 * \brief Keys, masks and filter number.
 */
typedef struct input_filter_key {
  uint key;     //!< MC packet key
  uint mask;    //!< MC packet mask
  uint filter;  //!< ID of filter to use for packets matching this key, mask
} input_filter_key_t;

/**
 * \brief Struct containing all input components.
 */
typedef struct ensemble_input {
  uint n_filters;     //!< Number of filters
  uint n_dimensions;  //!< Number of input dimensions for the ensemble
  uint n_routes;      //!< Number of input routing entries

  input_filter_key_t *routes;        //!< List of keys, masks, filter IDs
  filtered_input_buffer_t **filters; //!< Filters to apply to the inputs

  value_t *input;     //!< Resultant input value
} ensemble_input_t;

/* Buffers and parameters ****************************************************/
extern ensemble_input_t g_input;  //!< Input management

/* Functions *****************************************************************/

/**
 * \brief Initialise the input system
 * \param pars Formatted system region
 */
value_t* initialise_input( region_system_t *pars );

/**
 * \brief Handle an incoming dimensional value.
 * \param key Multicast key associated with the dimension
 * \param payload Partial value of the dimension to be accumulated
 *
 * Each arriving multicast packet contains a part of the value for a given
 * dimension for the given timestep.  On receipt of a packet the input
 * dimension referred to is taken from the bottom nibble of the key and the
 * value of the payload is added to the accumulator for this dimension.
 */
void incoming_dimension_value_callback( uint key, uint payload );

/**
 * \brief Handle the buffering and filtering of input
 *
 * Filter the inputs and set the accumulators to zero.
 */
void input_filter_step( void );

/**
 * \brief Return a pointer to the appropriate input filter for a given key.
 */
static inline filtered_input_buffer_t* input_filter( uint key ) {
  // Compare against each key, value pair held in the input
  for( uint i = 0; i < g_input.n_routes; i++ ) {
    if( ( key & g_input.routes[i].mask ) == g_input.routes[i].key ) {
      // Match the given key and mask
      return g_input.filters[ g_input.routes[i].filter ];
    }
  }
  // No match
  io_printf(IO_STD, "[Ensemble] ERROR Could not match incoming packet with key"
    " %d with filter.\n", key
  );
  return NULL;
};

#endif

/** @} */
