#include "filter.h"

filter_parameters_t g_filter;
uint delay_remaining;

void filter_update(uint arg0, uint arg1) {
  // Update the filters
  input_filter_step();

  io_printf(IO_STD, "[Filter] update\n");

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

void data_system(address_t addr) {
  g_filter.n_dimensions = addr[0];
  g_filter.machine_timestep = addr[1];
  g_filter.transmission_delay = addr[2];
  g_filter.n_filters = addr[3];
  g_filter.n_filter_keys = addr[4];

  delay_remaining = g_filter.transmission_delay;
  io_printf(IO_BUF, "[Filter] transmission delay = %d\n", delay_remaining);

  g_filter.input = initialise_input(
    g_filter.n_filters, g_filter.n_dimensions, g_filter.n_filter_keys);
}

void data_get_output_keys(address_t addr) {
  g_filter.keys = spin1_malloc(g_filter.n_dimensions * sizeof(uint));
  spin1_memcpy(
    g_filter.keys, addr, g_filter.n_dimensions * sizeof(uint));
}

void data_get_filters(address_t addr) {
  // TODO: Be less hacky
  for( uint f = 0; f < g_filter.n_filters; f++ ) {
    g_input.filters[f]->filter = kbits(addr[3*f + 0]);
    g_input.filters[f]->n_filter = kbits(addr[3*f + 1]);
    g_input.filters[f]->mask = addr[3*f + 2];
    g_input.filters[f]->mask_ = ~(addr[3*f + 2]);
  }
}

void data_get_filter_routing(address_t addr) {
  spin1_memcpy(
    g_input.routes, addr, g_input.n_routes * sizeof(input_filter_key_t));
}

void c_main(void) {
  address_t address = system_load_sram();
  data_system(region_start(1, address));
  data_get_output_keys(region_start(2, address));
  data_get_filters(region_start(3, address));
  data_get_filter_routing(region_start(4, address));

  // Set up routing tables
  if(leadAp) {
    system_lead_app_configured();
  }

  // Load core map
  system_load_core_map();

  // Setup timer tick, start
  spin1_set_timer_tick(g_filter.machine_timestep);
  spin1_callback_on(TIMER_TICK, filter_update, 2);
  spin1_start();
}
