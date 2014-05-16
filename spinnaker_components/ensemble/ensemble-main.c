#include "ensemble.h"

void c_main(void) {
  // Set the system up
  io_printf(IO_BUF, "[Ensemble] C_MAIN\n");
  address_t address = system_load_sram();
  if (!data_system(region_start(1, address))) {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }

  data_get_bias(region_start(2, address), g_ensemble.n_neurons);
  data_get_encoders(region_start(3, address), g_ensemble.n_neurons, g_input.n_dimensions);
  data_get_decoders(region_start(4, address), g_ensemble.n_neurons, g_n_output_dimensions);
  data_get_keys(region_start(5, address), g_n_output_dimensions);

  if (!get_filters(&g_input, region_start(6, address)) ||
      !get_filter_routes(&g_input, region_start(7, address))) {
    io_printf(IO_BUF, "[Ensemble] Failed to start.\n");
    return;
  }

  // Set up routing tables
  io_printf(IO_BUF, "[Ensemble] C_MAIN Configuring system.\n");
  if(leadAp){
    system_lead_app_configured();
  }

  // Setup timer tick, start
  io_printf(IO_BUF, "[Ensemble] C_MAIN Set timer and spin1_start.\n");
  spin1_set_timer_tick(g_ensemble.machine_timestep);
  spin1_start(SYNC_WAIT);
}
