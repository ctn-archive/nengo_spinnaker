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
input_filter_t g_input;
input_filter_t g_input_inhibitory;

/* Multicast Wrapper ********************************************************/
void mcpl_rx(uint key, uint payload) {
  if (!input_filter_mcpl_rx(&g_input, key, payload)) {
    if (!input_filter_mcpl_rx(&g_input_inhibitory, key, payload)) {
      io_printf(IO_BUF, "[MCPL Rx] Unknown key %08x\n", key);
    }
  }
}

/* Initialisation ***********************************************************/
bool initialise_ensemble(region_system_t *pars) {
  // Save constants
  g_ensemble.n_neurons = pars->n_neurons;
  g_ensemble.machine_timestep = pars->machine_timestep;
  g_ensemble.t_ref = pars->t_ref;
  g_ensemble.dt_over_t_rc = pars->dt_over_t_rc;
  g_ensemble.recd.record = pars->record_spikes;
  g_ensemble.n_inhib_dims = pars->n_inhibitory_dimensions;
  g_ensemble.inhib_gain = pars->inhibitory_gain;

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
  g_ensemble.input = input_filter_initialise(&g_input, pars->n_input_dimensions);
  if (g_ensemble.input == NULL)
    return false;

  if (NULL == input_filter_initialise(
        &g_input_inhibitory, pars->n_inhibitory_dimensions))
    return false;

  g_ensemble.output = initialise_output(pars);
  if (g_ensemble.output == NULL && g_n_output_dimensions > 0)
    return false;

  // Register the update function
  spin1_callback_on(TIMER_TICK, ensemble_update, 2);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_rx, -1);
  return true;
}
