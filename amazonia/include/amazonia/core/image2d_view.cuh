#pragma once

#include <amazonia/core/tags.cuh>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace amazonia
{
  template <typename T, typename D>
  class image2d_view;

  template <typename T>
  using image2d_view_host = image2d_view<T, host_t>;

  template <typename T>
  using image2d_view_device = image2d_view<T, device_t>;

  template <typename T, typename D>
  class image2d_view
  {
  public:
    using device_t = D;

  public:
    image2d_view() noexcept;
    image2d_view(const image2d_view&) noexcept;
    image2d_view(image2d_view&&) noexcept;
    image2d_view& operator=(const image2d_view&) noexcept;
    image2d_view& operator=(image2d_view&&) noexcept;

    image2d_view(T* buffer, int shapes[2], int strides[2]) noexcept;

    __host__ __device__ T&       operator()(int l, int c) noexcept;
    __host__ __device__ const T& operator()(int l, int c) const noexcept;

    __host__ __device__ int shape(int i) const noexcept;
    __host__ __device__ int nrows() const noexcept;
    __host__ __device__ int ncols() const noexcept;
    __host__ __device__ int stride(int i) const noexcept;

  protected:
    std::uint8_t* m_buffer;     // Buffer of the image
    int           m_shapes[2];  // Shapes of the image
    int           m_strides[2]; // Strides (in bytes) of the image
  };

  /*
   * Implementation
   */

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view() noexcept
    : m_buffer(nullptr)
  {
    std::memset(m_shapes, 0, 2 * sizeof(int));
    std::memset(m_shapes, 0, 2 * sizeof(int));
  }

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view(const image2d_view& other) noexcept
    : m_buffer(other.m_buffer)
  {
    std::memcpy(m_shapes, other.m_shapes, 2 * sizeof(int));
    std::memcpy(m_strides, other.m_strides, 2 * sizeof(int));
  }

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view(image2d_view&& other) noexcept
    : m_buffer(nullptr)
  {
    std::swap(m_buffer, other.m_buffer);
    std::memcpy(m_shapes, other.m_shapes, 2 * sizeof(int));
    std::memcpy(m_strides, other.m_strides, 2 * sizeof(int));
  }

  template <typename T, typename D>
  image2d_view<T, D>& image2d_view<T, D>::operator=(const image2d_view& other) noexcept
  {
    m_buffer = other.m_buffer;
    std::memcpy(m_shapes, other.m_shapes, 2 * sizeof(int));
    std::memcpy(m_strides, other.m_strides, 2 * sizeof(int));
    return *this;
  }

  template <typename T, typename D>
  image2d_view<T, D>& image2d_view<T, D>::operator=(image2d_view&& other) noexcept
  {
    m_buffer = std::exchange(other.m_buffer, nullptr);
    std::memcpy(m_shapes, other.m_shapes, 2 * sizeof(int));
    std::memcpy(m_strides, other.m_strides, 2 * sizeof(int));
    return *this;
  }

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view(T* buffer, int shapes[2], int strides[2]) noexcept
    : m_buffer(reinterpret_cast<std::uint8_t*>(buffer))
  {
    std::memcpy(m_shapes, shapes, 2 * sizeof(int));
    std::memcpy(m_strides, strides, 2 * sizeof(int));
  }

  template <typename T, typename D>
  __host__ __device__ T& image2d_view<T, D>::operator()(int l, int c) noexcept
  {
    assert(m_buffer);
    return *reinterpret_cast<T*>(m_buffer + m_strides[0] * l + m_strides[1] * c);
  }

  template <typename T, typename D>
  __host__ __device__ const T& image2d_view<T, D>::operator()(int l, int c) const noexcept
  {
    assert(m_buffer);
    return *reinterpret_cast<const T*>(m_buffer + m_strides[0] * l + m_strides[1] * c);
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::shape(int i) const noexcept
  {
    assert(i < 2);
    return m_shapes[i];
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::nrows() const noexcept
  {
    return shape(0);
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::ncols() const noexcept
  {
    return shape(1);
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::stride(int i) const noexcept
  {
    assert(i < 2);
    return m_strides[i];
  }
} // namespace amazonia