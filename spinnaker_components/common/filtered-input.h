/**
 * Filtered Input
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
 * \addtogroup filtered-input Filtered Input
 */

#ifndef __FILTERED_INPUT_H__
#define __FILTERED_INPUT_H__

#include "spin1_api.h"
#include "common-typedefs.h"
#include "nengo-common.h"

#include "dimensional-io.h"

/* Structs *******************************************************************/
/**
 * \brief Keys, masks and filter number.
 */
typedef struct input_filter_key {
  uint key;     //!< MC packet key
  uint mask;    //!< MC packet mask
  uint filter;  //!< ID of filter to use for packets matching this key, mask
  uint dimension_mask;  //!< Mask to retrieve dimension from key
} input_filter_key_t;


typedef struct input_filter_data {
  value_t filter;   //!< Filter value
  value_t filter_;  //!< 1.0 - filter value
  uint mask;        //!< Filter accumulator mask
  uint modulatory;  //!< Is this filter accumulatory
} input_filter_data_t;

/**
 * \brief Struct containing all input components.
 */
typedef struct filtered_input {
  uint n_filters;     //!< Number of filters
  uint n_dimensions;  //!< Number of input dimensions
  uint n_routes;      //!< Number of input routing entries

  input_filter_key_t *routes;        //!< List of keys, masks, filter IDs
  filtered_input_buffer_t **filters; //!< Filters to apply to the inputs

  value_t *input;     //!< Resultant input value
} filtered_input_t;

extern filtered_input_t g_input; //!< Global input

/* Functions *****************************************************************/

/**
 * \brief Initialise the input system
 */
value_t* initialise_input(uint n_input_dimensions);

/**
 * \brief Malloc sufficient space for filters and copy in filter parameters.
 * \param input input struct to which to add these filters.
 * \param filter_region address of filter region.
 */
bool get_filters(filtered_input_t* input, address_t filter_region);

/**
 * \brief Malloc sufficient space for the filter routers and copy in.
 * \param input input struct to which to add these filter routes.
 * \param routing_region address of filter routing region.
 */
bool get_filter_routes(filtered_input_t* input, address_t routing_region);

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

#endif

/** @} */
