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

  bool record;          //!< Whether or not to record the data in the buffer

  uint current_frame;   //!< Current frame number

  uint *_sdram_start;   //!< Start of the buffer in SDRAM
  uint *_sdram_current; //!< Current location in the SDRAM buffer
} recording_buffer_t;

/*!\brief Initialise a new recording buffer.
 */
bool record_buffer_initialise(recording_buffer_t *buffer, address_t region,
                              uint n_frames, uint n_neurons);

/*!\brief Prepare buffer for writing.
 *
 * If recording is in use the pointer to the current address in SDRAM will be
 * advanced by one recording frame.
 */
static inline void record_buffer_prepare(recording_buffer_t *buffer) {
  if (buffer->record) {
    buffer->_sdram_current += buffer->frame_length;
  }
}

/*!\brief Flush the current buffer.
 *
 * The contents of the buffer will be appended to the recording region in
 * SDRAM, but only if recording is in use.
 */
static inline void record_buffer_flush(recording_buffer_t *buffer) {
  // Copy the current buffer into SDRAM
  if (buffer->record) {
    spin1_memcpy(&buffer->_sdram_current, buffer->buffer,
                 buffer->frame_length * sizeof(uint));
  }
}

/*!\brief Record a spike for the given neuron.
 */
static inline void record_spike(recording_buffer_t *buffer, uint n_neuron) {
  // Get the offset within the current buffer, and the specific bit to set
  // We write to the buffer regardless of whether recording is desired or not
  // in order to reduce branching.
  buffer->buffer[n_neuron >> 5] |= 1 << (n_neuron & 0x1f);
}

#endif
