/*
 * Ensemble - Harness
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
 */

#include "ensemble.h"

/* Parameters and Buffers ***************************************************/
ensemble_parameters_t g_ensemble;

/* Initialisation ***********************************************************/
void initialise_ensemble(
  uint n_neurons,
  uint dt,
  uint n_input_dimensions,
  uint n_output_dimensions,
  uint t_ref,
  value_t one_over_t_rc,
  value_t filter
) {
  // Save constants
  g_ensemble.n_neurons = n_neurons;
  g_ensemble.machine_timestep = dt;
  g_ensemble.t_ref = t_ref;
  g_ensemble.one_over_t_rc = one_over_t_rc;

  // Holder for bias currents
  g_ensemble.i_bias = spin1_malloc(
    g_ensemble.n_neurons * sizeof( current_t )
  );

  // Holder for refactory period and voltages
  g_ensemble.status = spin1_malloc(
    g_ensemble.n_neurons * sizeof( uint )
  );
  for( uint n = 0; n < g_ensemble.n_neurons; n++ ){
    g_ensemble.status[n].refractory_time = 0;
    g_ensemble.status[n].voltage = 0;
  }

  // Initialise some buffers
  g_ensemble.encoders = spin1_malloc(
    g_ensemble.n_neurons * n_input_dimensions * sizeof( accum )
  );
  g_ensemble.decoders = spin1_malloc(
    g_ensemble.n_neurons * n_output_dimensions * sizeof( accum )
  );

  // Setup subcomponents
  initialise_input( n_input_dimensions, dt );
  initialise_output( n_output_dimensions, filter );
}
