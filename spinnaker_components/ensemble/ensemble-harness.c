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
bool initialise_ensemble(region_system_t *pars) {
  // Save constants
  g_ensemble.n_neurons = pars->n_neurons;
  g_ensemble.machine_timestep = pars->machine_timestep;
  g_ensemble.t_ref = pars->t_ref;
  g_ensemble.dt_over_t_rc = pars->dt_over_t_rc;

  io_printf(IO_BUF, "[Ensemble] INITIALISE_ENSEMBLE n_neurons = %d," \
            "timestep = %d, t_ref = %d, dt_over_t_rc = 0x%08x\n",
            g_ensemble.n_neurons,
            g_ensemble.machine_timestep,
            g_ensemble.t_ref,
            g_ensemble.dt_over_t_rc
  );

  // Holder for bias currents
  MALLOC_FAIL_FALSE(g_ensemble.i_bias,
                    g_ensemble.n_neurons * sizeof(current_t),
                    "[Ensemble]");

  // Holder for refractory period and voltages
  MALLOC_FAIL_FALSE(g_ensemble.status,
                    g_ensemble.n_neurons * sizeof(neuron_status_t),
                    "[Ensemble]");

  for (uint n = 0; n < g_ensemble.n_neurons; n++) {
    g_ensemble.status[n].refractory_time = 0;
    g_ensemble.status[n].voltage = 0;
  }

  // Initialise some buffers
  MALLOC_FAIL_FALSE(g_ensemble.encoders,
                    g_ensemble.n_neurons * pars->n_input_dimensions *
                      sizeof(value_t),
                    "[Ensemble]");

  MALLOC_FAIL_FALSE(g_ensemble.decoders,
                    g_ensemble.n_neurons * pars->n_output_dimensions *
                      sizeof(value_t),
                    "[Ensemble]");

  // Setup subcomponents
  g_ensemble.input = initialise_input(pars->n_input_dimensions);
  if (g_ensemble.input == NULL)
    return false;

  g_ensemble.output = initialise_output(pars);
  if (g_ensemble.output == NULL && g_n_output_dimensions > 0)
    return false;

  // Register the update function
  spin1_callback_on(TIMER_TICK, ensemble_update, 2);
  return true;
}
