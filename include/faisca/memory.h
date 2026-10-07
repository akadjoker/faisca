#pragma once

#include <cstddef>

namespace faisca
{

// a large block for weights: page aligned, huge pages when the system allows them
void* allocate_large(size_t bytes);

void free_large(void* block, size_t bytes);

}
