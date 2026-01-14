#pragma once

#include <amazonia/core/image2d.cuh>

#include <format>
#include <stdexcept>

namespace amazonia
{
  /// \brief Transfert host data from a 2D image view into a device 2D image view
  /// \param src The source host 2D image view
  /// \param dst The destination device 2D image view
  template <typename T>
  void transfer(const image2d_view<T, host_t>& src, image2d_view<T, device_t>& dst);

  /// \brief Transfert device data from a 2D image view into a host 2D image view
  /// \param src The source device 2D image view
  /// \param dst The destination host 2D image view
  template <typename T>
  void transfer(const image2d_view<T, device_t>&, image2d_view<T, host_t>&);

  /// \brief Transfert host data from a 2D image into device a 2D image
  /// \param src The source host 2D image
  /// \return A device 2D image
  template <typename T>
  image2d<T, device_t> transfer(const image2d_view<T, host_t>& src);

  /// \brief Transfert device data from a 2D image into a host 2D image
  /// \param src The source device 2D image
  /// \return A host 2D image
  template <typename T>
  image2d<T, host_t> transfer(const image2d_view<T, device_t>& src);

  /*
   * Implementations
   */

  template <typename T>
  void transfer(const image2d_view<T, host_t>& src, image2d_view<T, device_t>& dst)
  {
    assert(src.nrows() == dst.nrows() && src.ncols() == dst.ncols());
    const auto err = cudaMemcpy2D(dst.buffer(), dst.stride(0), src.buffer(), src.stride(0), src.ncols() * sizeof(T),
                                  src.nrows(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to transfert from host to device: {}", cudaGetErrorString(err)));
  }

  template <typename T>
  void transfer(const image2d_view<T, device_t>& src, image2d_view<T, host_t>& dst)
  {
    assert(src.nrows() == dst.nrows() && src.ncols() == dst.ncols());
    const auto err = cudaMemcpy2D(dst.buffer(), dst.stride(0), src.buffer(), src.stride(0), src.ncols() * sizeof(T),
                                  src.nrows(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to transfert from host to device: {}", cudaGetErrorString(err)));
  }

  template <typename T>
  image2d<T, device_t> transfer(const image2d_view<T, host_t>& src)
  {
    image2d<T, device_t> res(src.nrows(), src.ncols());
    transfer(src, res);
    return res;
  }

  template <typename T>
  image2d<T, host_t> transfer(const image2d_view<T, device_t>& src)
  {
    image2d<T, host_t> res(src.nrows(), src.ncols());
    transfer(src, res);
    return res;
  }
} // namespace amazonia