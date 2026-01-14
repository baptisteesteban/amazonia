#pragma once

#include <amazonia/core/tags.cuh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>

namespace amazonia
{
  /// \brief Class implementing a non owning data 2D image.
  /// \tparam T Data type of the image view values
  /// \tparam D Device location of the image view data
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

    /// \brief Constructor of an `image2d_view`
    /// \param buffer The input buffer. Its memory location (on host or device memory)
    /// is not checked and must be verified by the developper.
    /// \param shapes The shapes of the image in the form `{nrows, ncols}`.
    /// \param strides The amount of bytes to go to the next element at axis `i` in the iᵗʰ value of the table.

    image2d_view(T* buffer, int shapes[2], int strides[2]) noexcept;

    /// \brief Image value accessor (read/write)
    /// \param l The row of the desired value
    /// \param c The column of the desired value
    /// \return A reference to the desired value
    __host__ __device__ T& operator()(int l, int c) noexcept;

    /// \brief Image value accessor (read-only)
    /// \param l The row of the desired value
    /// \param c The column of the desired value
    /// \return A const reference to the desired value
    __host__ __device__ const T& operator()(int l, int c) const noexcept;

    /// \brief Shape information accessor
    /// \param i The desired shape axis
    /// \return The desired shape
    __host__ __device__ int shape(int i) const noexcept;

    /// \brief Get the number of row of the image
    __host__ __device__ int nrows() const noexcept;

    /// \brief Get the number of columns of the image
    __host__ __device__ int ncols() const noexcept;

    /// \brief Stride information accessor
    /// \param i The desired stride axis
    /// \return The desired stride
    __host__ __device__ int stride(int i) const noexcept;

    /// \brief Get the buffer of data (read/write)
    __host__ __device__ std::uint8_t* buffer() noexcept;

    /// \brief Get the buffer of data (read-only)
    __host__ __device__ const std::uint8_t* buffer() const noexcept;

  protected:
    std::uint8_t* m_buffer;     ///< Buffer of the image
    int           m_shapes[2];  ///< Shapes of the image
    int           m_strides[2]; ///< Strides (in bytes) of the image
  };

  /*
   * Implementation
   */

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view() noexcept
    : m_buffer(nullptr)
  {
    std::memset(m_shapes, 0, 2 * sizeof(int));
    std::memset(m_strides, 0, 2 * sizeof(int));
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
    assert(l >= 0 && c >= 0 && l < m_shapes[0] && c < m_shapes[1]);
    assert(m_buffer);
    return *reinterpret_cast<T*>(m_buffer + m_strides[0] * l + m_strides[1] * c);
  }

  template <typename T, typename D>
  __host__ __device__ const T& image2d_view<T, D>::operator()(int l, int c) const noexcept
  {
    assert(l >= 0 && c >= 0 && l < m_shapes[0] && c < m_shapes[1]);
    assert(m_buffer);
    return *reinterpret_cast<const T*>(m_buffer + m_strides[0] * l + m_strides[1] * c);
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::shape(int i) const noexcept
  {
    assert(i >= 0 && i < 2);
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
    assert(i >= 0 && i < 2);
    return m_strides[i];
  }

  template <typename T, typename D>
  __host__ __device__ std::uint8_t* image2d_view<T, D>::buffer() noexcept
  {
    return m_buffer;
  }

  template <typename T, typename D>
  __host__ __device__ const std::uint8_t* image2d_view<T, D>::buffer() const noexcept
  {
    return m_buffer;
  }
} // namespace amazonia