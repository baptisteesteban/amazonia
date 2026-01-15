#pragma once

#include <amazonia/core/image2d.cuh>

namespace amazonia::io
{
  /// \brief Read a 2D image from an image file.
  /// \param filename The input image filename.
  /// \param img The output image in which the input data will be stored
  template <typename T>
  void imread(const char* filename, image2d_host<T>& img);
} // namespace amazonia::io