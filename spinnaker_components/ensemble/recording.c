#include "recording.h"

bool record_buffer_initialise(recording_buffer_t *buffer, address_t region,
                              uint n_frames, uint n_neurons) {
  // Generate and store buffer parameters
  buffer->frame_length = (n_neurons >> 5) + (n_neurons & 0x1f ? 1 : 0);
  buffer->n_frames = n_frames;
  buffer->_sdram_buffer = region;

  buffer->current_frame = UINT32_MAX;  // To cause overflow on first tick

  // Create the local buffers
  MALLOC_FAIL_FALSE(buffer->_buffer_1, buffer->frame_length * sizeof(uint),
                    "[Recording]");
  MALLOC_FAIL_FALSE(buffer->_buffer_2, buffer->frame_length * sizeof(uint),
                    "[Recording]");
  buffer->buffer = buffer->_buffer_1;

  // Zero the local buffers
  for (uint i = 0; i < buffer->frame_length; i++) {
    buffer->_buffer_1[i] = 0x0;
    buffer->_buffer_2[i] = 0x0;
  }

  return true;
}
