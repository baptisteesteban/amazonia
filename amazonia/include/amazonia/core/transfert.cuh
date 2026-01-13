#pragma once

#include <amazonia/core/image2d.cuh>

#include <format>
#include <stdexcept>

namespace amazonia
{
  template <typename T>
  void transfert(const image2d_view<T, host_t>&, image2d_view<T, device_t>&);

  template <typename T>
  void transfert(const image2d_view<T, device_t>&, image2d_view<T, host_t>&);

  template <typename T>
  image2d<T, device_t> transfert(const image2d_view<T, host_t>&);

  template <typename T>
  image2d<T, host_t> transfert(const image2d_view<T, device_t>&);

  /*
   * Implementations
   */

  template <typename T>
  void transfert(const image2d_view<T, host_t>& src, image2d_view<T, device_t>& dst)
  {
    assert(src.nrows() == dst.nrows() && src.ncols() == dst.ncols());
    const auto err = cudaMemcpy2D(dst.buffer(), dst.stride(0), src.buffer(), src.stride(0), src.ncols(), src.nrows(),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to transfert from host to device: {}", cudaGetErrorString(err)));
  }

  template <typename T>
  void transfert(const image2d_view<T, device_t>& src, image2d_view<T, host_t>& dst)
  {
    assert(src.nrows() == dst.nrows() && src.ncols() == dst.ncols());
    const auto err = cudaMemcpy2D(dst.buffer(), dst.stride(0), src.buffer(), src.stride(0), src.ncols(), src.nrows(),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to transfert from host to device: {}", cudaGetErrorString(err)));
  }

  template <typename T>
  image2d<T, device_t> transfert(const image2d_view<T, host_t>& src)
  {
    image2d<T, device_t> res(src.nrows(), src.ncols());
    transfert(src, res);
    return res;
  }

  template <typename T>
  image2d<T, host_t> transfert(const image2d_view<T, device_t>& src)
  {
    image2d<T, host_t> res(src.nrows(), src.ncols());
    transfert(src, res);
    return res;
  }
} // namespace amazonia