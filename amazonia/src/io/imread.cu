#include <amazonia/io/imread.cuh>

#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>

#include "stb_image.h"

namespace amazonia::io
{
  template <>
  void imread(const char* filename, image2d_host<std::uint8_t>& img)
  {
    int           nrows, ncols, nchan;
    std::uint8_t* data = stbi_load(filename, &ncols, &nrows, &nchan, 0);
    if (!data)
      throw std::runtime_error(std::format("Unable to read image {}", filename));
    if (nchan != 1)
      throw std::runtime_error(std::format("Invalid image format (expected 1 channel, got {})", nchan));

    img.resize(nrows, ncols);
    std::memcpy(img.buffer(), data, nrows * ncols);
    stbi_image_free(data);
  }
} // namespace amazonia::io