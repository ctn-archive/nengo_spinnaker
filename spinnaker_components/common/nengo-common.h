#ifndef __NENGO_COMMON_H__
#define __NENGO_COMMON_H__

/*! \def MALLOC_FAIL_NULL
 *  Will malloc \a SIZE and save the returned pointer in \a VAR.  Should the
 *  malloc fail \a DESC will be printed to IO_BUF and NULL will be returned
 *  from the given function.
 */
#define MALLOC_FAIL_NULL(VAR, SIZE, DESC) \
do { VAR = spin1_malloc(SIZE); \
  if (VAR == NULL) { \
    io_printf(IO_BUF, DESC " Failed to malloc " #VAR " (%d bytes)\n", SIZE); \
    return NULL; \
} else { \
  io_printf(IO_BUF, DESC " Malloc " #VAR " (%d bytes)\n", SIZE); \
}} while (0)


/*! \def MALLOC_FAIL_FALSE
 *  Will malloc \a SIZE and save the returned pointer in \a VAR.  Should the
 *  malloc fail \a DESC will be printed to IO_BUF and FALSE will be returned
 *  from the given function.
 */
#define MALLOC_FAIL_FALSE(VAR, SIZE, DESC) \
do { VAR = spin1_malloc(SIZE); \
  if (VAR == NULL) { \
    io_printf(IO_BUF, DESC " Failed to malloc " #VAR " (%d bytes)\n", SIZE); \
    return false; \
} else { \
  io_printf(IO_BUF, DESC " Malloc " #VAR " (%d bytes)\n", SIZE); \
}} while(0)

#endif
