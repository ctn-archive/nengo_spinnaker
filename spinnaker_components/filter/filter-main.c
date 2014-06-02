#include "filter.h"

filter_parameters_t g_filter;
uint delay_remaining;
filtered_input_t g_input;

void filter_update(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Update the filters
  input_filter_step(&g_input);

  // Increment the counter and transmit if necessary
  delay_remaining--;
  if(delay_remaining == 0) {
    delay_remaining = g_filter.transmission_delay;

    uint val = 0x0000;
    for(uint d = 0; d < g_filter.n_dimensions; d++) {
      val = bitsk(g_filter.input[d]);
      spin1_send_mc_packet(g_filter.keys[d], val, WITH_PAYLOAD);
      io_printf(IO_STD, "[Filter] sent packet %d = %x\n", d, val);
    }
  }
}

bool data_system(address_t addr) {
  g_filter.n_dimensions = addr[0];
  g_filter.machine_timestep = addr[1];
  g_filter.transmission_delay = addr[2];

  delay_remaining = g_filter.transmission_delay;
  io_printf(IO_BUF, "[Filter] transmission delay = %d\n", delay_remaining);

  g_filter.input = initialise_input(&g_input, g_filter.n_dimensions);

  if (g_filter.input == NULL)
    return false;
  return true;
}

bool data_get_output_keys(address_t addr) {
  MALLOC_FAIL_FALSE(g_filter.keys,
                    g_filter.n_dimensions * sizeof(uint),
                    "[Filter]");
  spin1_memcpy(
    g_filter.keys, addr, g_filter.n_dimensions * sizeof(uint));

  return true;
}

void mcpl_callback(uint key, uint payload) {
  input_mcpl_rx(&g_input, key, payload);
}

void c_main(void) {
  address_t address = system_load_sram();
  if (!data_system(region_start(1, address)) ||
      !data_get_output_keys(region_start(2, address)) ||
      !get_filters(&g_input, region_start(3, address)) ||
      !get_filter_routes(&g_input, region_start(4, address))
  ) {
    io_printf(IO_BUF, "[Filter] Failed to initialise.\n");
    return;
  }

  // Set up routing tables
  if(leadAp) {
    system_lead_app_configured();
  }

  // Setup timer tick, start
  spin1_set_timer_tick(g_filter.machine_timestep);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mcpl_callback, -1);
  spin1_callback_on(TIMER_TICK, filter_update, 2);
  spin1_start(SYNC_WAIT);
}
