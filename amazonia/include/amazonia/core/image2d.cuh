#pragma once

#include <amazonia/core/image2d_data.cuh>
#include <amazonia/core/image2d_view.cuh>

#include <memory>

namespace amazonia
{
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

    image2d(int nrows, int ncols);

  private:
    std::shared_ptr<image2d_data<T, D>> m_data;
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
  image2d<T, D>::image2d(int nrows, int ncols)
    : m_data(std::make_shared<image2d_data<T, D>>(nrows, ncols))
  {
    this->m_buffer    = m_data->buffer;
    this->m_shapes[0] = nrows;
    this->m_shapes[1] = ncols;
    std::memcpy(this->m_strides, m_data->strides, 2 * sizeof(int));
  }
} // namespace amazonia