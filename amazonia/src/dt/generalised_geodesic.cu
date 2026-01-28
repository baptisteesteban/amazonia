#include <amazonia/dt/generalised_geodesic.cuh>

#include <cassert>
#include <cmath>

namespace amazonia::dt
{
  namespace
  {
    __constant__ float eucl_dist[3];

    __global__ void initialize_distance_transform(const image2d_view_device<std::uint8_t>& seeds,
                                                  image2d_view_device<float>&              out)
    {
      const int c = blockDim.x * blockIdx.x + threadIdx.x;
      const int l = blockDim.y * blockIdx.y + threadIdx.y;
      if (c < seeds.ncols() && l < seeds.nrows())
        out(l, c) = seeds(l, c) > 0 ? 0 : 1e10;
    }

    template <bool Forward>
    __global__ void pass(const image2d_view_device<std::uint8_t>& img, image2d_view_device<float>& out, int l_eucl,
                         int l_geos, bool* changed)
    {
      const int l = blockDim.x * blockIdx.x + threadIdx.x;
      if (l >= img.nrows())
        return;

      const int     start_c = Forward ? 1 : img.ncols() - 2;
      const int     end_c   = Forward ? img.ncols() : -1;
      constexpr int inc     = Forward ? 1 : -1;
      constexpr int dc      = -1 * inc;

      for (int c = start_c; c != end_c; c += inc)
      {
        const int nc = c + dc;
        for (int dl = -1; dl < 2; dl++)
        {
          const int nl = l + dl;
          if (nl < 0 || nl >= img.nrows())
            continue;

          const auto d = out(nl, nc) + l_eucl * eucl_dist[dl + 1] +
                         l_geos * (img(l, c) < img(nl, nc) ? img(nl, nc) - img(l, c) : img(l, c) - img(nl, nc));
          if (d < out(l, c))
          {
            out(l, c) = d;
            *changed  = true;
          }
        }
      }
    }

    template <bool Forward>
    __global__ void pass_T(const image2d_view_device<std::uint8_t>& img, image2d_view_device<float>& out, int l_eucl,
                           int l_geos, bool* changed)
    {
      const int c = blockDim.x * blockIdx.x + threadIdx.x;
      if (c >= img.ncols())
        return;

      const int     start_l = Forward ? 1 : img.nrows() - 2;
      const int     end_l   = Forward ? img.nrows() : -1;
      constexpr int inc     = Forward ? 1 : -1;
      constexpr int dl      = -1 * inc;

      for (int l = start_l; l != end_l; l += inc)
      {
        const int nl = l + dl;
        for (int dc = -1; dc < 2; dc++)
        {
          const int nc = c + dc;
          if (nc < 0 || nc >= img.ncols())
            continue;

          const auto d = out(nl, nc) + l_eucl * eucl_dist[dc + 1] +
                         l_geos * (img(l, c) < img(nl, nc) ? img(nl, nc) - img(l, c) : img(l, c) - img(nl, nc));
          if (d < out(l, c))
          {
            out(l, c) = d;
            *changed  = true;
          }
        }
      }
    }
  } // namespace

  /// \brief Compute the generalised geodesic distance transform on a 2D image. Be careful to provide an `out` image
  /// whose dimension are the same as `img` and `mask`.
  /// \param img The input image.
  /// \param seeds The seed points from which the distance transform is computed. It is represented as an image whose
  /// values is > 0 is the pixel point is a seed and 0 otherwise.
  /// \param out The output distance map.
  /// \param lambda The distance transform parameter. If lambda is 0, the distance is Euclidean; If lambda is 1, the
  /// distance is geodesic, else, it is a mix of both.
  void generalised_geodesic(const image2d_view_device<std::uint8_t>& img,
                            const image2d_view_device<std::uint8_t>& seeds, image2d_view_device<float>& out,
                            float lambda)
  {
    assert(img.nrows() == seeds.nrows() && img.ncols() == seeds.ncols());
    assert(img.nrows() == out.nrows() && img.ncols() == out.ncols());

    const int l_eucl = 1 - lambda;

    // Initialize constant memory
    float local_dist[] = {sqrtf(2), 1, sqrtf(2)};
    cudaMemcpyToSymbol(eucl_dist, local_dist, 3 * sizeof(float));

    // Initialize the output distance map according to the seed points
    constexpr int BLOCK_SIZE = 32;
    dim3          grid_dim((img.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (img.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    {
      dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
      initialize_distance_transform<<<grid_dim, block_dim>>>(seeds, out);
    }

    bool* changed;
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;
    while (*changed)
    {
      *changed = false;
      pass<true><<<grid_dim.y, BLOCK_SIZE>>>(img, out, l_eucl, lambda, changed);
      pass<false><<<grid_dim.y, BLOCK_SIZE>>>(img, out, l_eucl, lambda, changed);
      pass_T<true><<<grid_dim.x, BLOCK_SIZE>>>(img, out, l_eucl, lambda, changed);
      pass_T<false><<<grid_dim.x, BLOCK_SIZE>>>(img, out, l_eucl, lambda, changed);
      cudaDeviceSynchronize();
    }
    cudaFree(changed);
    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(std::format("[generalised_geodesic_2d] error: {}", cudaGetErrorString(err)));
  }

  /// \brief Compute the generalised geodesic distance transform.
  /// \param img The input image.
  /// \param seeds The seed points from which the distance transform is computed.
  /// It is represented as an image whose values is > 0 is the pixel point is
  /// a seed and 0 otherwise.
  /// \param lambda The distance transform parameter. If lambda is 0, the distance is Euclidean; If lambda is 1, the
  /// distance is geodesic, else, it is a mix of both.
  /// \return The output distance map.
  image2d_device<float> generalised_geodesic(const image2d_view_device<std::uint8_t>& img,
                                             const image2d_view_device<std::uint8_t>& seeds, float lambda)
  {
    assert(img.nrows() == seeds.nrows() && img.ncols() == seeds.ncols());
    image2d_device<float> out(img.nrows(), img.ncols());
    generalised_geodesic(img, seeds, out, lambda);
    return out;
  }
} // namespace amazonia::dt