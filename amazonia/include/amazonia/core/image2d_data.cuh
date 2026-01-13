#pragma once

#include <amazonia/core/tags.cuh>

#include <concepts>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace amazonia
{
  template <typename T, typename D>
  struct image2d_data final
  {
    static_assert(std::same_as<D, host_t> || std::same_as<D, device_t>);
    image2d_data()                               = delete;
    image2d_data(const image2d_data&)            = delete;
    image2d_data(image2d_data&&)                 = delete;
    image2d_data& operator=(const image2d_data&) = delete;
    image2d_data& operator=(image2d_data&&)      = delete;

    image2d_data(int nrows, int ncols);
    ~image2d_data();

    std::uint8_t* buffer;
    int           strides[2];
  };

  /*
   * Implementations
   */

  template <typename T, typename D>
  image2d_data<T, D>::image2d_data(int nrows, int ncols)
    : buffer(nullptr)
  {
    constexpr std::size_t e_size = sizeof(T);
    std::size_t           pitch;
    if constexpr (std::same_as<D, host_t>)
    {
      buffer = (std::uint8_t*)std::malloc(nrows * ncols * e_size);
      pitch  = ncols * e_size;
      if (!buffer)
        throw std::runtime_error(std::format("Error while allocating host data"));
    }
    else
    {
      auto e = cudaMallocPitch(&buffer, &pitch, ncols * e_size, nrows);
      if (e != cudaSuccess)
        throw std::runtime_error(std::format("Error while allocating device data: {}", cudaGetErrorString(e)));
    }
    strides[0] = pitch;
    strides[1] = e_size;
  }

  template <typename T, typename D>
  image2d_data<T, D>::~image2d_data()
  {
    if constexpr (std::same_as<D, host_t>)
      std::free(buffer);
    else
      cudaFree(buffer);
  }
} // namespace amazonia