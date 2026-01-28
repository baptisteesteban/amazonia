#include <amazonia/core/rgb.cuh>
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
    int           width, height, nchan;
    std::uint8_t* data = stbi_load(filename, &width, &height, &nchan, 0);
    if (!data)
      throw std::runtime_error(std::format("Unable to read image {}", filename));
    if (nchan != 1)
      throw std::runtime_error(std::format("Invalid image format (expected 1 channel, got {})", nchan));

    img.resize(width, height);
    std::memcpy(img.buffer(), data, width * height);
    stbi_image_free(data);
  }

  template <>
  void imread(const char* filename, image2d_host<rgb8>& img)
  {
    int           width, height, nchan;
    std::uint8_t* data = stbi_load(filename, &width, &height, &nchan, 0);
    if (!data)
      throw std::runtime_error(std::format("Unable to read image {}", filename));
    if (nchan != 3)
      throw std::runtime_error(std::format("Invalid image format (expected 3 channels, got {})", nchan));

    img.resize(width, height);
    std::memcpy(img.buffer(), data, width * height * sizeof(rgb8));
    stbi_image_free(data);
  }
} // namespace amazonia::io