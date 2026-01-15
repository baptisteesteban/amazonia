#pragma once

#include <amazonia/core/image2d.cuh>

namespace amazonia::dt
{
  /// \brief Compute the generalised geodesic distance transform on a 2D image.
  /// Be careful to provide an `out` image whose dimension are the same as `img`
  /// and `mask`.
  /// \param img The input image.
  /// \param seeds The seed points (pixels whose value is greater than 0).
  /// \param out The output image.
  /// \param lambda The generalised geodesic distance parameter
  void generalised_geodesic(const image2d_view_device<std::uint8_t>& img,
                            const image2d_view_device<std::uint8_t>& seeds, image2d_view_device<float>& out,
                            float lambda);

  image2d_device<float> generalised_geodesic(const image2d_view_device<std::uint8_t>& img,
                                             const image2d_view_device<std::uint8_t>& seeds, float lambda);
} // namespace amazonia::dt