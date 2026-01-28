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
      const int x = blockDim.x * blockIdx.x + threadIdx.x;
      const int y = blockDim.y * blockIdx.y + threadIdx.y;
      if (x < seeds.width() && y < seeds.height())
        out(x, y) = seeds(x, y) > 0 ? 0 : 1e10;
    }

    template <bool Forward>
    __global__ void pass(const image2d_view_device<std::uint8_t>& img, image2d_view_device<float>& out, float l_eucl,
                         float l_geos, bool* changed)
    {
      const int y = blockDim.x * blockIdx.x + threadIdx.x;
      if (y >= img.height())
        return;

      const int     start_x = Forward ? 1 : img.width() - 2;
      const int     end_x   = Forward ? img.width() : -1;
      constexpr int inc     = Forward ? 1 : -1;
      constexpr int dx      = -1 * inc;

      for (int x = start_x; x != end_x; x += inc)
      {
        const int nx = x + dx;
        for (int dy = -1; dy < 2; dy++)
        {
          const int ny = y + dy;
          if (ny < 0 || ny >= img.height())
            continue;

          const auto d = out(nx, ny) + l_eucl * eucl_dist[dy + 1] +
                         l_geos * (img(x, y) < img(nx, ny) ? img(nx, ny) - img(x, y) : img(x, y) - img(nx, ny));
          if (d < out(x, y))
          {
            out(x, y) = d;
            *changed  = true;
          }
        }
      }
    }

    template <bool Forward>
    __global__ void pass_T(const image2d_view_device<std::uint8_t>& img, image2d_view_device<float>& out, float l_eucl,
                           float l_geos, bool* changed)
    {
      const int x = blockDim.x * blockIdx.x + threadIdx.x;
      if (x >= img.width())
        return;

      const int     start_y = Forward ? 1 : img.height() - 2;
      const int     end_y   = Forward ? img.height() : -1;
      constexpr int inc     = Forward ? 1 : -1;
      constexpr int dy      = -1 * inc;

      for (int y = start_y; y != end_y; y += inc)
      {
        const int ny = y + dy;
        for (int dx = -1; dx < 2; dx++)
        {
          const int nx = x + dx;
          if (nx < 0 || nx >= img.width())
            continue;

          const auto d = out(nx, ny) + l_eucl * eucl_dist[dx + 1] +
                         l_geos * (img(x, y) < img(nx, ny) ? img(nx, ny) - img(x, y) : img(x, y) - img(nx, ny));
          if (d < out(x, y))
          {
            out(x, y) = d;
            *changed  = true;
          }
        }
      }
    }
  } // namespace

  void generalised_geodesic(const image2d_view_device<std::uint8_t>& img,
                            const image2d_view_device<std::uint8_t>& seeds, image2d_view_device<float>& out,
                            float lambda)
  {
    assert(img.width() == seeds.width() && img.height() == seeds.height());
    assert(img.width() == out.width() && img.height() == out.height());

    const float l_eucl = 1.0f - lambda;

    // Initialize constant memory
    float local_dist[] = {sqrtf(2), 1, sqrtf(2)};
    cudaMemcpyToSymbol(eucl_dist, local_dist, 3 * sizeof(float));

    // Initialize the output distance map according to the seed points
    constexpr int BLOCK_SIZE = 32;
    dim3          grid_dim((img.width() + BLOCK_SIZE - 1) / BLOCK_SIZE, (img.height() + BLOCK_SIZE - 1) / BLOCK_SIZE);
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

  image2d_device<float> generalised_geodesic(const image2d_view_device<std::uint8_t>& img,
                                             const image2d_view_device<std::uint8_t>& seeds, float lambda)
  {
    assert(img.width() == seeds.width() && img.height() == seeds.height());
    image2d_device<float> out(img.width(), img.height());
    generalised_geodesic(img, seeds, out, lambda);
    return out;
  }
} // namespace amazonia::dt