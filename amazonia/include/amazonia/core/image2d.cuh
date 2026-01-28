#pragma once

#include <amazonia/core/image2d_data.cuh>
#include <amazonia/core/image2d_view.cuh>

#include <memory>

namespace amazonia
{
  /// \brief Class implementing a owning data 2D image. This class inherits from `image2d_view`
  /// so it uses its methods.
  /// \tparam T Data type of the image values
  /// \tparam D Device location of the image data
  template <typename T, typename D>
  class image2d;

  template <typename T>
  using image2d_host = image2d<T, host_t>;

  template <typename T>
  using image2d_device = image2d<T, device_t>;

  template <typename T, typename D>
  class image2d final : public image2d_view<T, D>
  {
  public:
    using image2d_view<T, D>::device_t;

  public:
    image2d() noexcept;
    image2d(const image2d&) noexcept;
    image2d(image2d&&) noexcept;
    image2d& operator=(const image2d&) noexcept;
    image2d& operator=(image2d&&) noexcept;

    /// \brief Constructor of a 2D image.
    /// \param width The width of the desired image.
    /// \param height The height of the desired image.
    image2d(int width, int height);

    /// \brief Resize the internal storage of a 2D image. Be careful: the
    /// resize operation drop the data stored in the image and replace it
    /// by a new buffer
    /// \param width The desired width
    /// \param height The desired height
    void resize(int width, int height);

  private:
    std::shared_ptr<image2d_data<T, D>> m_data; ///< Data storage
  };

  /*
   * Implementations
   */

  template <typename T, typename D>
  image2d<T, D>::image2d() noexcept
    : image2d_view<T, D>()
    , m_data(nullptr)
  {
  }

  template <typename T, typename D>
  image2d<T, D>::image2d(const image2d& other) noexcept
    : image2d_view<T, D>(other)
    , m_data(other.m_data)
  {
  }

  template <typename T, typename D>
  image2d<T, D>::image2d(image2d&& other) noexcept
    : image2d_view<T, D>(other)
  {
    m_data = std::exchange(m_data, nullptr);
  }

  template <typename T, typename D>
  image2d<T, D>& image2d<T, D>::operator=(const image2d& other) noexcept
  {
    image2d_view<T, D>::operator=(other);
    m_data = other.m_data;
    return *this;
  }

  template <typename T, typename D>
  image2d<T, D>& image2d<T, D>::operator=(image2d&& other) noexcept
  {
    image2d_view<T, D>::operator=(other);
    m_data = std::exchange(other.m_data, nullptr);
    return *this;
  }

  template <typename T, typename D>
  image2d<T, D>::image2d(int width, int height)
    : m_data(std::make_shared<image2d_data<T, D>>(width, height))
  {
    this->m_buffer = m_data->buffer;
    this->m_width  = width;
    this->m_height = height;
    this->m_spitch = m_data->spitch;
    this->m_epitch = m_data->epitch;
  }

  template <typename T, typename D>
  void image2d<T, D>::resize(int width, int height)
  {
    auto candidate = std::make_shared<image2d_data<T, D>>(width, height);
    if (candidate)
    {
      m_data         = candidate;
      this->m_buffer = m_data->buffer;
      this->m_width  = width;
      this->m_height = height;
      this->m_spitch = m_data->spitch;
      this->m_epitch = m_data->epitch;
    }
  }
} // namespace amazonia