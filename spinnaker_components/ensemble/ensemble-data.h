#include "common-impl.h"

#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__

bool copy_in_system_region( uint *addr ); // Copy in global data
bool copy_in_bias( uint *addr );          // Copy in bias data
bool copy_in_encoders( uint *addr );      // Copy in encoders
bool copy_in_decoders( uint *addr );      // Copy in decoders
bool copy_in_decoder_keys( uint *addr );  // Copy in decoder keys

#endif
