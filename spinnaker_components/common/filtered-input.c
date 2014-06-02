/*
 * Filtered Input
 * --------------
 * Structures and functions to deal with arriving multicast packets (input).
 *
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 */

#include "filtered-input.h"

value_t* initialise_input(filtered_input_t* input, uint n_input_dimensions) {
  input->n_dimensions = n_input_dimensions;

  MALLOC_FAIL_NULL(input->input,
                   input->n_dimensions * sizeof(value_t),
                   "[Common/Input]");

  // Return the input (to the encoders) buffer
  return input->input;
}

// Filter initialisation
bool get_filters(filtered_input_t* input, address_t filter_region) {
  input->n_filters = filter_region[0];

  io_printf(IO_BUF, "[Filters] n_filters = %d, n_input_dimensions = %d\n",
            input->n_filters, input->n_dimensions);

  if (input->n_filters > 0) {
    MALLOC_FAIL_FALSE(input->filters,
                      input->n_filters * sizeof(filtered_input_buffer_t*),
                      "[Common/Input]");

    input_filter_data_t* filters = (input_filter_data_t*) (filter_region + 1);

    for (uint f = 0; f < input->n_filters; f++) {
      input->filters[f] = input_buffer_initialise(input->n_dimensions);
      input->filters[f]->filter = filters[f].filter;
      input->filters[f]->n_filter = filters[f].filter_;
      input->filters[f]->mask = filters[f].mask;
      input->filters[f]->mask_ = ~filters[f].mask;

      io_printf(IO_BUF, "Filter [%d] = %k/%k Masked: 0x%08x/0x%08x\n",
                f, filters[f].filter, filters[f].filter_, filters[f].mask,
                ~filters[f].mask);
    };
  }

  return true;
}

// Filter routers initialisation
bool get_filter_routes(filtered_input_t* input, address_t routing_region) {
  input->n_routes = routing_region[0];

  io_printf(IO_BUF, "[Common/Input] %d filter routes.\n", input->n_routes);

  if (input->n_filters > 0 && input->n_routes > 0) {
    MALLOC_FAIL_FALSE(input->routes,
                      input->n_routes * sizeof(input_filter_key_t),
                      "[Common/Input]");
    spin1_memcpy(input->routes, routing_region + 1, 
                 input->n_routes * sizeof(input_filter_key_t));

    for (uint r = 0; r < input->n_routes; r++) {
      io_printf(IO_BUF,
                "Filter route [%d] 0x%08x && 0x%08x => %d with dmask 0x%08x\n",
                r, input->routes[r].key, input->routes[r].mask,
                input->routes[r].filter, input->routes[r].dimension_mask);
    }
  }

  return true;
}

// Incoming spike callback
void input_mcpl_rx(filtered_input_t* input, uint key, uint payload) {
  /*
   * 1. Look up key in input routing table entry
   * 2. Select appropriate filter
   * 3. Add value (payload) to appropriate dimension of given filter.
   */
  // Compare against each key, value pair held in the input
  for (uint i = 0; i < input->n_routes; i++) {
    if ((key & input->routes[i].mask ) == input->routes[i].key) {
      input_buffer_acc(input->filters[input->routes[i].filter],
                       key & input->routes[i].dimension_mask,
                       kbits(payload));
      return;
    }
  }

  // No match
  io_printf(IO_STD, "[Filtered Input] ERROR Could not match incoming packet "
    "with key %d with filter.\n", key
  );
}

// Input step
void input_filter_step(filtered_input_t* input) {
  // Zero the input accumulator
  for (uint d = 0; d < input->n_dimensions; d++) {
    input->input[d] = 0x00000000;
  }

  // For each filter, apply filtering and accumulate the value in the global
  // input accumulator.
  for (uint f = 0; f < input->n_filters; f++) {
    input_buffer_step(input->filters[f]);

    for (uint d = 0; d < input->n_dimensions; d++) {
      input->input[d] += input->filters[f]->filtered[d];
    }
  }
}
