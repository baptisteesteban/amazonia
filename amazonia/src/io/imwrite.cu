#include <amazonia/core/rgb.cuh>
#include <amazonia/io/imwrite.cuh>

#include <format>
#include <stdexcept>
#include <string_view>

#include "stb_image_write.h"

namespace amazonia::io
{
  template <>
  void imwrite(const char* filename, const image2d_view_host<std::uint8_t>& img)
  {
    if (!std::string_view(filename).ends_with(".png"))
      throw std::invalid_argument(std::format("Only PNG file can be saved (Got filename {})", filename));

    stbi_write_png(filename, img.width(), img.height(), 1, img.buffer(), img.spitch());
  }

  template <>
  void imwrite(const char* filename, const image2d_view_host<rgb8>& img)
  {
    if (!std::string_view(filename).ends_with(".png"))
      throw std::invalid_argument(std::format("Only PNG file can be saved (Got filename {})", filename));

    stbi_write_png(filename, img.width(), img.height(), 3, img.buffer(), img.spitch());
  }
} // namespace amazonia::io