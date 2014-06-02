#include "value_sink.h"

address_t rec_start, rec_curr;
uint n_dimensions;
value_t *input;

void sink_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Filter inputs, write the latest value to SRAM
  input_filter_step();
  spin1_memcpy(rec_curr, input, n_dimensions * sizeof(value_t));
  rec_curr = &rec_curr[n_dimensions];
}

void c_main(void)
{
  // Load SRAM, lead application
  address_t address = system_load_sram();
  if (leadAp) {
    system_lead_app_configured();
  }

  // Load parameters and filters
  region_system_t *pars = (region_system_t *) region_start(1, address);
  n_dimensions = pars->n_dimensions;
  input = initialise_input(n_dimensions);

  if (input == NULL) {
    return;
  }

  if (!get_filters(&g_input, region_start(2, address)) ||
      !get_filter_routes(&g_input, region_start(3, address))
  ) {
    io_printf(IO_BUF, "[Value Sink] Failed to start.\n");
    return;
  }
  rec_start = rec_curr = region_start(15, address);

  // Set up callbacks, start
  spin1_set_timer_tick(pars->timestep);
  spin1_callback_on(TIMER_TICK, sink_update, 2);
  spin1_start(SYNC_WAIT);
}
