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
extern uint g_n_input_dimensions;    //!< Number of input dimensions \f$D_{in}\f$

extern value_t * gp_ibuf_accumulator;//!< Input buffers \f$1 \times D_{in}\f$
extern value_t * gp_ibuf_filtered;   //!< Filtered input buffers \f$1 \times D_{in}\f$

extern value_t g_filter;             //!< Input decay factor

/* Functions *****************************************************************/

/**
 * \brief Initialise the input system
 * \param n Number of input dimensions \f$D_{in}\f$
 * \param f Value of the input filter
 */
void initialise_input( uint n, value_t f );

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
  for( uint d = 0; d < g_n_input_dimensions; d++ ) {
    gp_ibuf_filtered[d] *= g_filter;
    gp_ibuf_filtered[d] += gp_ibuf_accumulator[d] *
      ( (value_t) 1.0k - g_filter );

    gp_ibuf_accumulator[d] = 0x00000000;
  }
}

#endif

/** @} */
