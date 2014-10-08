#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "common-impl.h"

#include "sim_controller.h"

// Reverse direction of timer to ensure time is increasing
#define NOW (-tc2[TC_COUNT])

typedef struct _keys_t {
  uint ping;
  uint pong;
  uint correction;
} keys_t;

// Number of nodes to send out pings/corrections to
static uint num_nodes;

// An array of num_nodes routing key definitions
static keys_t *keys;

// Keys for multicatsing a start or stop command
static uint start_at_key;
static uint stop_at_key;

// The keys for communicating with the currently in-progress node.
static volatile keys_t cur_node_keys;

// The value of tc2 when the last ping was sent
static volatile uint ping_send_time;

// Was a pong received?
static volatile uint pong_received;

// The correction sent to the last node
static volatile int last_correction;

// The range of corrections applied during the last round of corrections.
static volatile uint correction_range = -1;


/*!
 * \brief Respond to pings with corrections.
 */
static void on_mcpl_rx(uint key, uint remote_time) {
  uint pong_recv_time = NOW;
  
  if (key != cur_node_keys.pong) {
    io_printf(IO_BUF, "[Sim Controller] Got unexpected pong with key %08x, expected %08x.\n",
              key, cur_node_keys.pong);
    return;
  }
  
  // Estimate the node's clock time
	uint latency = (pong_recv_time - ping_send_time)/2;
	remote_time += latency;
	
	// Send a relative correction
	uint error = (uint)(((int)pong_recv_time) - ((int)remote_time));
	spin1_send_mc_packet(cur_node_keys.correction, error, WITH_PAYLOAD);
	
	last_correction = error;
	pong_received = TRUE;
}


/*!
 * \brief Send a reply to a query command received over SDP.
 */
static void send_sdp_reply(sim_control_cmd_t command, uint response) {
  sdp_msg_t message;
  
  // Send to 0,0
  message.dest_addr = 0x0000;
  message.dest_port = 0xFF;
  message.srce_addr = sv->p2p_addr;
  message.srce_port = spin1_get_id();
  
  // No reply required
  message.flags = 0x07;
  
  // Will be ejected on IP tag 2.
  message.tag = 2;
  
  // The meat of the reply
  message.cmd_rc = (ushort)command;
  message.arg1 = response;
  
  // No arguments
  message.length = sizeof(sdp_hdr_t) + sizeof(cmd_hdr_t);
  
  spin1_send_sdp_msg(&message, 1000);
}


/*!
 * \brief Handle incoming commands from the host
 */
static void on_sdp_rx(uint mailbox, uint port) {
  use(port);
  
  sdp_msg_t *message = (sdp_msg_t*) mailbox;
  
  sim_control_cmd_t command = (sim_control_cmd_t)message->cmd_rc;
  
  switch (command) {
    case SIM_CTL_GET_TIME:
      send_sdp_reply(SIM_CTL_GET_TIME, NOW);
      break;
    
    case SIM_CTL_GET_DRIFT:
      send_sdp_reply(SIM_CTL_GET_DRIFT, correction_range);
      break;
    
    case SIM_CTL_START_AT:
      spin1_send_mc_packet(start_at_key, message->arg1, WITH_PAYLOAD);
      break;
    
    case SIM_CTL_STOP_AT:
      spin1_send_mc_packet(stop_at_key, message->arg1, WITH_PAYLOAD);
      break;
    
    default:
      io_printf(IO_BUF, "[Sim Controller] Got unexpected command via SDP: %d\n",
                command);
  }
  
  spin1_msg_free(message);
}


/*!
 * \brief Send out pings to remote simulation nodes.
 */
static void on_timer_tick(uint arg1, uint arg2) {
  use(arg1);
  use(arg2);
  
  static uint cur_node = -1;
  
  static int min_correction;
  static int max_correction;
  
  if (!pong_received && cur_node != (uint)-1) {
    io_printf(IO_BUF, "[Sim Controller] Node %d did not respond in time to ping with key %d!\n",
              cur_node, cur_node_keys.ping);
  } else if (pong_received && cur_node != (uint)-1) {
    if (last_correction < min_correction || cur_node == 0)
      min_correction = last_correction;
    if (last_correction > max_correction || cur_node == 0)
      max_correction = last_correction;
  }
  
  if (++cur_node >= num_nodes) {
    cur_node = 0;
    correction_range = (uint)(max_correction - min_correction);
   }
  cur_node_keys = keys[cur_node];
  
  spin1_send_mc_packet(cur_node_keys.ping, 0, WITH_PAYLOAD);
  ping_send_time = NOW;
  pong_received = FALSE;
}


void c_main(void) {
  // Set the system up
  io_printf(IO_BUF, "[Sim Controller] C_MAIN\n");
  address_t address = system_load_sram();
  
  // Get pointers to the list of routing keys to use (note that these are read
  // directly from SDRAM since use is infrequent).
  num_nodes = ((uint *)region_start(1, address))[0];
  keys = (keys_t *)(((uint *)region_start(1, address))[1]);
  
  // Configure reference timer appropriately
  tc2[TC_CONTROL] = ( (0 << 0) /* Wrapping counter */
                    | (1 << 1) /* 32-bit counter */
                    | (0 << 2) /* Clock divider (/1 = 0, /16 = 1, /256 = 2) */
                    | (0 << 5) /* No interrupt */
                    | (0 << 6) /* Free-running */
                    | (1 << 7) /* Enabled */
                    );
  
  // Setup callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, on_mcpl_rx, -1);
  spin1_callback_on(SDP_PACKET_RX, on_sdp_rx, 0);
  spin1_callback_on(TIMER_TICK, on_timer_tick, 1);
  
  // Go!
  spin1_start(SYNC_WAIT);
}
