#ifndef SRC_CORE_COMMON_H
#define SRC_CORE_COMMON_H

namespace arboretum {
namespace core {
union my_atomics {
  float floats[2];               // floats[0] = maxvalue
  unsigned int ints[2];          // ints[1] = maxindex
  unsigned long long int ulong;  // for atomic update
};
}  // namespace core
}  // namespace arboretum
#endif