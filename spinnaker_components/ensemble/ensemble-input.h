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

/* Buffers and parameters ****************************************************/
extern uint g_n_input_dimensions;            //!< Number of input dimensions
                                             //   \f$D_{in}\f$
extern filtered_input_buffer_t *gfib_input;  //!< Input buffer

/* Functions *****************************************************************/

/**
 * \brief Initialise the input system
 * \param n Number of input dimensions \f$D_{in}\f$
 * \param pars Formatted system region
 */
void initialise_input( uint n, region_system_t *pars );

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
static inline void input_filter_step( void ) {
  input_buffer_step( gfib_input );
}

#endif

/** @} */
