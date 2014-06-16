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

#include "ensemble-pes.h"

#include <string.h>

//----------------------------------
// Structs
//----------------------------------
// Structure defining structure of PES region
// **TODO** this could be an opaque type here
struct region_pes_t
{
  // Scalar learning rate (scaled by dt) used in PES decoder delta calculation
  value_t learning_rate;

  // Values defining the decay of the exponential filter applied 
  // To the neuron activity used in PES decoder delta calculation
  value_t activity_decay;
  
  // Index of the input signal filter that contains error signal
  uint error_signal_filter_index;
  
  // Offset into decoder to apply PES
  uint decoder_output_offset;
};

//----------------------------------
// Global variables
//----------------------------------
pes_parameters_t g_pes;

//----------------------------------
// Global functions
//----------------------------------
void get_pes(struct region_pes_t *pars)
{
  // Setup global PES state from region
  g_pes.learning_rate = pars->learning_rate;
  g_pes.activity_decay = pars->activity_decay;
  g_pes.error_signal_filter_index = pars->error_signal_filter_index;
  g_pes.decoder_output_offset = pars->decoder_output_offset;
  
  io_printf(IO_BUF, "PES learning: Learning rate:%k, Activity decay:%k, Error signal filter index:%u, Decoder output offset:%u\n", 
            g_pes.learning_rate, g_pes.activity_decay, g_pes.error_signal_filter_index, g_pes.decoder_output_offset);
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
/*void pes_update_neuron_activity(uint n, bool spiked)
{
  // If learning is enabled
  if(g_pes.learning_rate > 0.0k)
  {
    // Decay neuron's filtered activity
    g_pes.filtered_activity[n] *= g_pes.activity_decay;
    
    // If neuron's spiked, add extra energy into trace
    if(spiked)
    {
      g_pes.filtered_activity[n] += 1.0k;
    }
  }
}
//----------------------------------
void pes_update()
{
  // If PES learning is enabled
  if(g_pes.learning_rate > 0.0k)
  {
    // Extract error signal vector from 
    const value_t *filtered_error_signal = g_input.filters[g_pes.error_signal_filter_index]->filtered;
    
    // Loop through output dimensions
    for(uint n = 0; n < g_ensemble.n_neurons; n++) 
    {
      // Get filtered activity of this neuron and it's decoder vector
      value_t filtered_activity = g_pes.filtered_activity[n];
      value_t *decoder_vector = neuron_decoder_vector(n);
      
      // Loop through output dimensions and apply PES to decoder values offset by output offset
      for(uint d = 0; d < g_input.n_dimensions; d++) 
      {
        decoder_vector[d + g_pes.decoder_output_offset] += (g_pes.learning_rate * filtered_activity * filtered_error_signal[d]);
      }
    }
  }
}*/