#pragma once

#include <amazonia/core/image2d_view.cuh>

namespace amazonia::io
{
  /// \brief Save an image into a file
  /// \param filename The output filename
  /// \param img The source image to be saved into a file
  template <typename T>
  void imwrite(const char* filename, const image2d_view_host<T>& img);
} // namespace amazonia::io