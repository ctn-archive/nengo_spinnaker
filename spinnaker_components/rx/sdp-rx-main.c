#include "sdp-rx.h"

sdp_rx_parameters_t g_sdp_rx;

/** \brief Timer tick
 */
void sdp_rx_tick(uint arg0, uint arg1) {
  spin1_send_mc_packet(g_sdp_rx.keys[g_sdp_rx.current_dimension],
                       bitsk(g_sdp_rx.output[g_sdp_rx.current_dimension]),
                       WITH_PAYLOAD);

  g_sdp_rx.current_dimension++;
  if (g_sdp_rx.current_dimension >= g_sdp_rx.n_dimensions) {
    g_sdp_rx.current_dimension = 0;
  }
}

/** \brief Receive packed data packed in SDP message
 */
void sdp_received(uint mailbox, uint port) {
  sdp_msg_t *message = (sdp_msg_t*) mailbox;

  // Copy the data into the output buffer
  spin1_memcpy(g_sdp_rx.output, message->data,
               g_sdp_rx.n_dimensions * sizeof(value_t));

  spin1_msg_free(message);
}

/** \brief Load in system parameters
 */
void data_system(address_t addr) {
  g_sdp_rx.transmission_period = addr[0];
  g_sdp_rx.n_dimensions = addr[1];

  io_printf(IO_BUF, "[SDP Rx] Transmission period: %d\n",
            g_sdp_rx.transmission_period);
  io_printf(IO_BUF, "[SDP Rx] %d dimensions.\n", g_sdp_rx.n_dimensions);

  g_sdp_rx.output = spin1_malloc(g_sdp_rx.n_dimensions * sizeof(value_t));
  g_sdp_rx.keys = spin1_malloc(g_sdp_rx.n_dimensions * sizeof(uint));
}

/** \brief Load output keys
 */
void data_get_keys(address_t addr) {
  spin1_memcpy(g_sdp_rx.keys, addr, g_sdp_rx.n_dimensions * sizeof(uint));

  for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
    io_printf(IO_BUF, "[SDP Rx] Key[%2d] = 0x%08x\n", d, g_sdp_rx.keys[d]);
  }
}

/** \brief Main function
 */
void c_main(void) {
  address_t address = system_load_sram();
  data_system(region_start(1, address));
  data_get_keys(region_start(2, address));

  g_sdp_rx.current_dimension = 0;

  for (uint d = 0; d < g_sdp_rx.n_dimensions; d++) {
    g_sdp_rx.output[d] = 0x00000000;
  }

  // Set up routing tables
  if(leadAp) {
    system_lead_app_configured();
  }

  // Load core map
  system_load_core_map();

  // Setup timer tick, start
  spin1_set_timer_tick(g_sdp_rx.transmission_period);
  spin1_callback_on(SDP_PACKET_RX, sdp_received, 0);
  spin1_callback_on(TIMER_TICK, sdp_rx_tick, 2);
  spin1_start();
}
