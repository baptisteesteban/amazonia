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
    /// \param width The width of the view.
    /// \param height The height of the view.
    /// \param spitch The pitch (in bytes) between two contiguous lines.
    /// \param epitch The pitch (in bytes) between two contiguous elements.
    image2d_view(T* buffer, int width, int height, int spitch, int epitch = sizeof(T)) noexcept;

    /// \brief Image value accessor (read/write)
    /// \param x The `x`-coordinate of the desired pixel value
    /// \param y The `y`-coordinate of the desired pixel value
    /// \return A reference to the desired value
    __host__ __device__ T& operator()(int x, int y) noexcept;

    /// \brief Image value accessor (read-only)
    /// \param x The `x`-coordinate of the desired pixel value
    /// \param y The `y`-coordinate of the desired pixel value
    /// \return A const reference to the desired value
    __host__ __device__ const T& operator()(int x, int y) const noexcept;

    /// \brief Get the width of the image
    __host__ __device__ int width() const noexcept;

    /// \brief Get the height of the image
    __host__ __device__ int height() const noexcept;

    /// \brief Get the pitch (in bytes) between two contiguous lines of an image
    __host__ __device__ int spitch() const noexcept;

    /// \brief Get the pitch (in bytes) between two contiguous values of an image
    __host__ __device__ int epitch() const noexcept;

    /// \brief Get the buffer of data (read/write)
    __host__ __device__ std::uint8_t* buffer() noexcept;

    /// \brief Get the buffer of data (read-only)
    __host__ __device__ const std::uint8_t* buffer() const noexcept;

  protected:
    std::uint8_t* m_buffer; ///< Buffer of the image.
    int           m_width;  ///< The width of the image.
    int           m_height; ///< The height of the image.
    int           m_spitch; ///< The pitch (in bytes) from a line of the image to the other.
    int           m_epitch; ///< The pitch (in bytes) between two contiguous value on a row.
  };

  /*
   * Implementation
   */

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view() noexcept
    : m_buffer(nullptr)
    , m_width(0)
    , m_height(0)
    , m_spitch(0)
    , m_epitch(0)
  {
  }

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view(const image2d_view& other) noexcept
    : m_buffer(other.m_buffer)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_spitch(other.m_spitch)
    , m_epitch(other.m_epitch)
  {
  }

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view(image2d_view&& other) noexcept
    : m_buffer(nullptr)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_spitch(other.m_spitch)
    , m_epitch(other.m_epitch)
  {
    std::swap(m_buffer, other.m_buffer);
  }

  template <typename T, typename D>
  image2d_view<T, D>& image2d_view<T, D>::operator=(const image2d_view& other) noexcept
  {
    m_buffer = other.m_buffer;
    m_width  = other.m_width;
    m_height = other.m_height;
    m_spitch = other.m_spitch;
    m_epitch = other.m_epitch;
    return *this;
  }

  template <typename T, typename D>
  image2d_view<T, D>& image2d_view<T, D>::operator=(image2d_view&& other) noexcept
  {
    m_buffer = std::exchange(other.m_buffer, nullptr);
    m_buffer = other.m_buffer;
    m_width  = other.m_width;
    m_height = other.m_height;
    m_spitch = other.m_spitch;
    m_epitch = other.m_epitch;
    return *this;
  }

  template <typename T, typename D>
  image2d_view<T, D>::image2d_view(T* buffer, int width, int height, int spitch, int epitch) noexcept
    : m_buffer(reinterpret_cast<std::uint8_t*>(buffer))
    , m_width(width)
    , m_height(height)
    , m_spitch(spitch)
    , m_epitch(epitch)
  {
  }

  template <typename T, typename D>
  __host__ __device__ T& image2d_view<T, D>::operator()(int x, int y) noexcept
  {
    assert(x >= 0 && y >= 0 && x < m_width && y < m_height);
    assert(m_buffer);
    return *reinterpret_cast<T*>(m_buffer + m_spitch * y + m_epitch * x);
  }

  template <typename T, typename D>
  __host__ __device__ const T& image2d_view<T, D>::operator()(int x, int y) const noexcept
  {
    assert(x >= 0 && y >= 0 && x < m_width && y < m_height);
    assert(m_buffer);
    return *reinterpret_cast<const T*>(m_buffer + m_spitch * y + m_epitch * x);
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::width() const noexcept
  {
    return m_width;
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::height() const noexcept
  {
    return m_height;
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::spitch() const noexcept
  {
    return m_spitch;
  }

  template <typename T, typename D>
  __host__ __device__ int image2d_view<T, D>::epitch() const noexcept
  {
    return m_epitch;
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