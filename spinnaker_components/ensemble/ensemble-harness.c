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
void initialise_ensemble( region_system_t *pars ) {
  // Save constants
  g_ensemble.n_neurons = pars->n_neurons;
  g_ensemble.machine_timestep = pars->machine_timestep;
  g_ensemble.t_ref = pars->t_ref;
  g_ensemble.dt_over_t_rc = pars->dt_over_t_rc;

  io_printf( IO_BUF, "[Ensemble] INITIALISE_ENSEMBLE n_neurons = %d," \
             "timestep = %d, t_ref = %d, dt_over_t_rc = 0x%08x\n",
             g_ensemble.n_neurons,
             g_ensemble.machine_timestep,
             g_ensemble.t_ref,
             g_ensemble.dt_over_t_rc
  );

  // Holder for bias currents
  io_printf( IO_BUF, "[Ensemble] INITIALISE malloc-ing i_bias.\n" );
  g_ensemble.i_bias = spin1_malloc(
    g_ensemble.n_neurons * sizeof( current_t )
  );

  // Holder for refractory period and voltages
  io_printf( IO_BUF, "[Ensemble] INITIALISE malloc-ing neuron status.\n" );
  g_ensemble.status = spin1_malloc(
    g_ensemble.n_neurons * sizeof( neuron_status_t )
  );
  for( uint n = 0; n < g_ensemble.n_neurons; n++ ){
    g_ensemble.status[n].refractory_time = 0;
    g_ensemble.status[n].voltage = 0;
  }

  // Initialise some buffers
  io_printf( IO_BUF, "[Ensemble] INITIALISE malloc-ing encoders.\n" );
  g_ensemble.encoders = spin1_malloc(
    g_ensemble.n_neurons * pars->n_input_dimensions * sizeof( value_t )
  );
  io_printf( IO_BUF, "[Ensemble] INITIALISE malloc-ing decoders.\n" );
  g_ensemble.decoders = spin1_malloc(
    g_ensemble.n_neurons * pars->n_output_dimensions * sizeof( value_t )
  );

  // Setup subcomponents
  g_ensemble.input  = initialise_input(
    pars->n_filters, pars->n_input_dimensions);
  g_ensemble.output = initialise_output( pars );

  // Register the update function
  spin1_callback_on( TIMER_TICK, ensemble_update, 2 );
}
