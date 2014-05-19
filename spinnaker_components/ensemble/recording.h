/*!
 * \brief Spike recording
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 *
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 */

#ifndef __RECORDING_H__
#define __RECORDING_H__

#include "spin1_api.h"
#include "common-typedefs.h"
#include "nengo-common.h"

typedef struct _recording_buffer_t {
  uint *buffer;         //!< The buffer to write to
  uint frame_length;    //!< Size of 1 frame of the buffer (in words)
  uint n_frames;        //!< Length of the buffer in frames (= n_ticks)

  uint current_frame;   //!< Current frame number

  uint *_sdram_buffer;  //!< Location of the buffer in SDRAM

  uint *_buffer_1;      //!< Pointer to first buffer
  uint *_buffer_2;      //!< Pointer to second buffer
} recording_buffer_t;

/*!\brief Initialise a new recording buffer.
 */
bool record_buffer_initialise(recording_buffer_t *buffer, address_t region,
                              uint n_frames, uint n_neurons);

/*!\brief Prepare buffer for writing.
 */
static inline void record_buffer_prepare(recording_buffer_t *buffer) {
  // Point to next buffer and clear
  buffer->buffer = (buffer->buffer == buffer->_buffer_1) ?
                    buffer->_buffer_2 : buffer->_buffer_1;

  for (uint i = 0; i < buffer->frame_length; i++) {
    buffer->buffer[i] = 0x0;
  }

  buffer->current_frame++;
}

/*!\brief Flush the current buffer.
 */
static inline void record_buffer_flush(recording_buffer_t *buffer) {
  // Copy the current buffer into SDRAM
  spin1_memcpy(
    &buffer->_sdram_buffer[buffer->current_frame * buffer->frame_length],
    buffer->buffer,
    buffer->frame_length * sizeof(uint)
  );
}

/*!\brief Record a spike for the given neuron.
 */
static inline void record_spike(recording_buffer_t *buffer, uint n_neuron) {
  // Get the offset within the current buffer, and the specific bit to set
  buffer->buffer[n_neuron >> 5] |= 1 << (n_neuron & 0x1f);
}

#endif
