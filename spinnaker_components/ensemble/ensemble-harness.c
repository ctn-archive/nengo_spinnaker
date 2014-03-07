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
  region_system_t *pars
) {
  // Save constants
  g_ensemble.n_neurons = pars->n_neurons;
  g_ensemble.machine_timestep = pars->machine_timestep;
  g_ensemble.t_ref = pars->t_ref;
  g_ensemble.one_over_t_rc = pars->one_over_t_rc;

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
    g_ensemble.n_neurons * pars->n_input_dimensions * sizeof( accum )
  );
  g_ensemble.decoders = spin1_malloc(
    g_ensemble.n_neurons * pars->n_output_dimensions * sizeof( accum )
  );

  // Setup subcomponents
  initialise_input( pars->n_input_dimensions, pars->machine_timestep );
  initialise_output( pars->n_output_dimensions, pars->filter );
}
