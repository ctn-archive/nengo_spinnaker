/*
 * Ensemble - Data
 *
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 * 
 */

#include "ensemble.h"

#include <string.h>

//----------------------------------
// Global variables
//----------------------------------
pes_parameters_t g_pes;

//----------------------------------
// Global functions
//----------------------------------
void get_pes(region_pes_t *pars)
{
  // Setup global PES state from region
  g_pes.learning_rate = pars->learning_rate;
  g_pes.activity_decay = pars->activity_decay;
  g_pes.one_minus_activity_decay = 1.0k - g_pes.activity_decay;
  
  io_printf(IO_BUF, "PES learning: Learning rate:%k, Activity decay:%k", g_pes.learning_rate, g_pes.activity_decay);
}
//----------------------------------
bool initialise_pes(uint n_neurons)
{
  // Allocate memory for filtered neuron activity
  MALLOC_FAIL_FALSE(g_pes.filtered_activity, n_neurons * sizeof(value_t),
    "[PES]");
  
  // Zero it
  memset(g_pes.filtered_activity, 0, n_neurons * sizeof(value_t));
  
  return true;
}
//----------------------------------
void pes_update(uint error_input_filter)
{
  // If PES learning is enabled
  if(g_pes.learning_rate > 0.0k)
  {
    // Extract error signal vector from 
    const value_t *filtered_error_signal = g_input.filters[error_input_filter]->filtered;
    
    // Loop through output dimensions
    for(uint n = 0; n < g_ensemble.n_neurons; n++) 
    {
      // Get filtered activity of this neuron and it's decoder vector
      value_t filtered_activity = g_pes.filtered_activity[n];
      value_t *decoder_vector = neuron_decoder_vector(n);
      
      // Loop through output dimensions and apply PES to decoder values
      for(uint d = 0; d < g_n_output_dimensions; d++) 
      {
        decoder_vector[d] += (g_pes.learning_rate * filtered_activity * filtered_error_signal[d]);
      }
    }
  }
}