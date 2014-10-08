#include "value_source.h"
#include "slots.h"

slots_t slots;            // Slots for output data
uint* keys;               // Output keys
system_parameters_t pars; // Global system parameters
uint n_blocks;            // Number of blocks (in total)
uint current_block;       // Current block
uint8_t* blocks;          // Location of blocks in DRAM

void valsource_tick(uint ticks, uint arg1) {
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks) {
    spin1_exit(0);
  }

  // Transmit a MC packet for each value in the current frame
  for (uint d = 0; d < pars.n_dims; d++) {
    spin1_send_mc_packet(
        keys[d],
        slots.current->data[slots.current->current_pos*pars.n_dims + d],
        WITH_PAYLOAD
    );
  }

  // Copy in the next block
  if (slots.current->current_pos == 0) {
    if (n_blocks > 1) {
      // More than one block, need to copy in subsequent block
      void *s_addr = (void*) &blocks[(current_block + 1) * pars.block_length *
                                     pars.n_dims * sizeof(value_t)];

      if (current_block == pars.n_blocks - 1) {
        // Subsequent block is the LAST block
        spin1_dma_transfer(0, s_addr, slots.next->data, DMA_READ,
                           pars.partial_block * pars.n_dims * sizeof(value_t));
        slots.next->length = pars.partial_block;
      } else if (current_block == pars.n_blocks) {
        // Current block is the LAST block
        if (pars.flags & 0x1) {
          // We are wrapping, so next block is the FIRST block
          spin1_dma_transfer(0, blocks, slots.next->data, DMA_READ,
                             pars.block_length * pars.n_dims * sizeof(value_t));
          slots.next->length = pars.block_length;
        }
      } else {
        // Nothing special about subsequent block
        spin1_dma_transfer(0, s_addr, slots.next->data, DMA_READ,
                           pars.block_length * pars.n_dims * sizeof(value_t));
        slots.next->length = pars.block_length;
      }
    }
  }

  // Switch blocks if necessary
  slots.current->current_pos++;
  if (slots.current->current_pos == slots.current->length) {
    // We've reached the end of the current slot, progress or wrap
    if (n_blocks == 1) {
      // Only one block: wrap or exit
      if (pars.flags & 0x1) {
        // Function is periodic: wrap to start
        slots.current->current_pos = 0;
      } else {
        // Function is not periodic: exit
        spin1_exit(0);
      }
    } else {
      // Multiple blocks: next, wrap or exit
      if (current_block == n_blocks - 1 && !(pars.flags & 0x1)) {
        // Last block, aperiodic: exit
        spin1_exit(0);
      } else {
        // Not last block, or periodic: next
        slots_progress(&slots);
        current_block++;

        // Wrap if necessary
        if (current_block == n_blocks)
          current_block = 0;
      }
    }
  }
}

void c_main(void) {
  address_t address = system_load_sram();
  if (leadAp) {
    system_lead_app_configured();
  }

  // Copy in the system region
  spin1_memcpy(&pars, region_start(1, address), sizeof(system_parameters_t));
  n_blocks = pars.n_blocks + (pars.partial_block > 0 ? 1 : 0);
  current_block = 0;
  blocks = (void *) region_start(3, address);

  io_printf(IO_BUF, "[Value Source] %d dimensions, %d full blocks of %d FRAMES"
                    ", PLUS %d FRAMES = %d blocks\n",
            pars.n_dims, pars.n_blocks, pars.block_length, pars.partial_block,
            n_blocks);

  // Make space for keys
  keys = spin1_malloc(pars.n_dims * sizeof(uint));
  if (keys == NULL) {
    io_printf(IO_BUF, "Failed to malloc space for keys.\n");
    return;
  }
  spin1_memcpy(keys, region_start(2, address), pars.n_dims * sizeof(uint));

  // Initialise the slots with 20KB buffer space
  if (!initialise_slots(&slots, 20*1024)) {
    return;
  }

  // Copy in the first block of data
  if (n_blocks > 1) {
    spin1_memcpy(slots.current->data, region_start(3, address),
                 pars.n_dims * pars.block_length * sizeof(value_t));
    slots.current->length = pars.block_length;
  } else {
    spin1_memcpy(slots.current->data, region_start(3, address),
                 pars.n_dims * pars.partial_block * sizeof(value_t));
    slots.current->length = pars.partial_block;
  }

  // Set up callbacks, wait for synchronisation
  spin1_set_timer_tick(pars.time_step);
  spin1_callback_on(TIMER_TICK, valsource_tick, 0);
  spin1_start(SYNC_WAIT);
}
