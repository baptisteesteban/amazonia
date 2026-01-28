#pragma once

#include <amazonia/core/tags.cuh>

#include <concepts>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace amazonia
{

  /// @brief 2D image storage for element type `T` in domain `D`.
  /// @tparam T Element type (for example `std::uint8_t`, `float`).
  /// @tparam D Device tag — must be `host_t` or `device_t`.
  ///
  /// The class owns a pitched buffer and frees it on destruction. The
  /// buffer is stored as a raw `std::uint8_t*` to keep the layout
  /// flexible; element access should be performed using `strides` elements.
  template <typename T, typename D>
  struct image2d_data final
  {
    static_assert(std::same_as<D, host_t> || std::same_as<D, device_t>);

    image2d_data()                               = delete;
    image2d_data(const image2d_data&)            = delete;
    image2d_data(image2d_data&&)                 = delete;
    image2d_data& operator=(const image2d_data&) = delete;
    image2d_data& operator=(image2d_data&&)      = delete;


    /// @brief Construct and allocate a pitched 2D buffer.
    /// @param width The width of the buffer.
    /// @param height The height of the buffer.
    /// @throws std::runtime_error on allocation failure.
    ///
    /// For `host_t` tag, a contiguous host allocation is performed via
    /// `std::malloc`. For `device_t` tag, `cudaMallocPitch` is used and the
    /// returned `pitch` becomes `spitch` (bytes per row).
    image2d_data(int width, int height);
    ~image2d_data();

    std::uint8_t* buffer;
    int           spitch;
    int           epitch;
  };

  /*
   * Implementations
   */

  template <typename T, typename D>
  image2d_data<T, D>::image2d_data(int width, int height)
    : buffer(nullptr)
  {
    constexpr std::size_t e_size = sizeof(T);
    std::size_t           pitch;
    if constexpr (std::same_as<D, host_t>)
    {
      buffer = (std::uint8_t*)std::malloc(width * height * e_size);
      pitch  = width * e_size;
      if (!buffer)
        throw std::runtime_error(std::format("Error while allocating host data"));
    }
    else
    {
      auto e = cudaMallocPitch(&buffer, &pitch, width * e_size, height);
      if (e != cudaSuccess)
        throw std::runtime_error(std::format("Error while allocating device data: {}", cudaGetErrorString(e)));
    }
    spitch = pitch;
    epitch = e_size;
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