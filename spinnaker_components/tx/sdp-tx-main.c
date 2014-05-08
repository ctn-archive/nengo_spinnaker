#include "sdp-tx.h"

sdp_tx_parameters_t g_sdp_tx;
uint delay_remaining;

void sdp_tx_update(uint ticks, uint arg1) {
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Update the filters
  input_filter_step();

  // Increment the counter and transmit if necessary
  delay_remaining--;
  if(delay_remaining == 0) {
    delay_remaining = g_sdp_tx.transmission_delay;

    // Construct and transmit the SDP Message
    sdp_msg_t message;
    message.dest_addr = 0x0000;        // (0, 0)
    message.dest_port = 0xff;
    message.srce_addr = sv->p2p_addr;  // Sender P2P address
    message.srce_port = spin1_get_id();
    message.flags = 0x07;              // No reply expected
    message.tag = 1;                   // Send to IPtag 1

    message.cmd_rc = 1;
    spin1_memcpy(
      message.data, g_sdp_tx.input, g_sdp_tx.n_dimensions * sizeof(value_t));

    message.length = sizeof(sdp_hdr_t) + sizeof(cmd_hdr_t) + 
                     g_sdp_tx.n_dimensions * sizeof(value_t);

    spin1_send_sdp_msg(&message, 100);
  }
}

void data_system(address_t addr) {
  g_sdp_tx.n_dimensions = addr[0];
  g_sdp_tx.machine_timestep = addr[1];
  g_sdp_tx.transmission_delay = addr[2];
  g_sdp_tx.n_filters = addr[3];
  g_sdp_tx.n_filter_keys = addr[4];

  delay_remaining = g_sdp_tx.transmission_delay;
  io_printf(IO_BUF, "[SDP Tx] Tick period = %d microseconds\n",
            g_sdp_tx.machine_timestep);
  io_printf(IO_BUF, "[SDP Tx] transmission delay = %d\n", delay_remaining);

  g_sdp_tx.input = initialise_input(
    g_sdp_tx.n_filters, g_sdp_tx.n_dimensions, g_sdp_tx.n_filter_keys);
}

void data_get_filters(address_t addr) {
  // TODO: Be less hacky
  for( uint f = 0; f < g_sdp_tx.n_filters; f++ ) {
    g_input.filters[f]->filter = kbits(addr[3*f + 0]);
    g_input.filters[f]->n_filter = kbits(addr[3*f + 1]);
    g_input.filters[f]->mask = addr[3*f + 2];
    g_input.filters[f]->mask_ = ~(addr[3*f + 2]);

    io_printf(IO_BUF, "Filter[%d] = %k, %k\n", f,
      g_input.filters[f]->filter,
      g_input.filters[f]->n_filter
    );
  }
}

void data_get_filter_routing(address_t addr) {
  spin1_memcpy(
    g_input.routes, addr, g_input.n_routes * sizeof(input_filter_key_t));
}

void c_main(void) {
  address_t address = system_load_sram();
  data_system(region_start(1, address));
  data_get_filters(region_start(2, address));
  data_get_filter_routing(region_start(3, address));

  // Set up routing tables
  if(leadAp) {
    system_lead_app_configured();
  }

  // Setup timer tick, start
  spin1_set_timer_tick(g_sdp_tx.machine_timestep);
  spin1_callback_on(TIMER_TICK, sdp_tx_update, 2);
  spin1_start(SYNC_WAIT);
}
