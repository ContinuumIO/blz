#include <blosc.h>

#if BLOSC_VERSION_MAJOR == 1 && BLOSC_VERSION_MINOR < 3
#error "Blosc version >= 1.3 required"
#endif
