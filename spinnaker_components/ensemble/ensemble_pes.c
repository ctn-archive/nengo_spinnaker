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

#include "ensemble_pes.h"

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
  g_pes.error_signal_filter_index = pars->error_signal_filter_index;
  g_pes.decoder_output_offset = pars->decoder_output_offset;
  
  io_printf(IO_BUF, "PES learning: Learning rate:%k, Error signal filter index:%u, Decoder output offset:%u\n", 
            g_pes.learning_rate, g_pes.error_signal_filter_index, g_pes.decoder_output_offset);
}