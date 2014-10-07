#include "spin1_api.h"

#include "common-typedefs.h"

#include "discipline.h"
#include "disciplined_clock.h"
#include "disciplined_timer.h"


// Keep a copy of the MC keys
static region_discipline_keys_t keys;

// On the first correction, just update the phase
static bool first_correction;

// The timestep for timer interrupts (ticks)
static uint timestep;


void discipline_initialise(address_t pars, uint timestep_usec) {
  keys = *((region_discipline_keys_t *)pars);
  first_correction = true;
  timestep = sv->cpu_clk * timestep_usec;
  dclk_initialise_state();
}


bool discipline_process_mc_packet(uint key, uint payload) {
  if (key == keys.ping ) {
    // Respond to pings with the current time
    spin1_send_mc_packet(keys.pong, dclk_read_raw_time(), TRUE);
    return true;
  } else if (key == keys.correction ) {
    // Apply corrections from the sim controller
    if (first_correction)
      dclk_correct_phase_now(payload);
    else
      dclk_add_correction(payload);
    first_correction = false;
    return true;
  } else if (key == keys.start_at ) {
    // Set the time at which interrupts will start
    dtimer_start_interrupts(payload, timestep);
    return true;
  } else if (key == keys.stop_at ) {
    // Set the time at which interrupts will stop
    dtimer_stop_interrupts(payload);
    return true;
  } else {
    return false;
  }
}
