/**
 * Ensemble - PES
 * -----------------
 * Functions to perform PES decoder learning
 * 
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 * 
 * \addtogroup ensemble
 * @{
 */


#ifndef __ENSEMBLE_PES_H__
#define __ENSEMBLE_PES_H__

#include "ensemble.h"

//----------------------------------
// Forward declarations
//----------------------------------
struct region_pes_t;

//----------------------------------
// Structs
//----------------------------------
// Structure containing parameters and state required for PES learning
typedef struct pes_parameters_t
{
  // Scalar learning rate used in PES decoder delta calculation
  value_t learning_rate;
  
  // Index of the input signal filter that contains error signal
  uint error_signal_filter_index;
  
  // Offset into decoder to apply PES
  uint decoder_output_offset;
} pes_parameters_t;

//----------------------------------
// External variables
//----------------------------------
extern pes_parameters_t g_pes;

//----------------------------------
// Inline functions
//----------------------------------
/**
* \brief When using non-filtered activity, applies PES when neuron spikes
*/
static inline void pes_neuron_spiked(uint n)
{
  if(g_pes.learning_rate > 0.0k)
  {
    // Extract error signal vector from 
    const value_t *filtered_error_signal = g_input_modulatory.filters[g_pes.error_signal_filter_index]->filtered;
    
    // Get filtered activity of this neuron and it's decoder vector
    value_t *decoder_vector = neuron_decoder_vector(n);
    
    // Loop through output dimensions and apply PES to decoder values offset by output offset
    for(uint d = 0; d < g_input.n_dimensions; d++) 
    {
      decoder_vector[d + g_pes.decoder_output_offset] += (g_pes.learning_rate * filtered_error_signal[d]);
    }
  }
}

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the PES learning 
* rule to the PES region of the Ensemble.
*/
void get_pes(struct region_pes_t *pars);

/** @} */

#endif  // __ENSEMBLE_PES_H__