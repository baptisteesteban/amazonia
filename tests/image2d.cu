#include <amazonia/core/image2d.cuh>
#include <amazonia/core/transfer.cuh>

#include <gtest/gtest.h>

TEST(image2d, u8_host)
{
  using namespace amazonia;

  std::uint8_t data[] = {2, 8, 9, 4, 3, 6};

  image2d_host<std::uint8_t> img(3, 2);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.spitch(), 3);
  ASSERT_EQ(img.epitch(), 1);

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      img(x, y) = data[y * 3 + x];
  }

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      ASSERT_EQ(img(x, y), data[y * 3 + x]);
  }
}

TEST(image2d, u32_host)
{
  using namespace amazonia;
  using value_t        = std::uint32_t;
  constexpr int e_size = sizeof(value_t);

  value_t data[] = {2, 8, 9, 4, 3, 6};

  image2d_host<value_t> img(3, 2);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.spitch(), 3 * e_size);
  ASSERT_EQ(img.epitch(), e_size);

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      img(x, y) = data[y * 3 + x];
  }

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      ASSERT_EQ(img(x, y), data[y * 3 + x]);
  }

  img.resize(3, 2);
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.spitch(), 3 * e_size);
  ASSERT_EQ(img.epitch(), e_size);
}

template <typename T>
__global__ void add_one(amazonia::image2d_view<T, amazonia::device_t>& img)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < img.width() && y < img.height())
    img(x, y) += 1;
}

TEST(image2d, u8_device)
{
  using namespace amazonia;

  std::uint8_t       data[]       = {2, 8, 9, 4, 3, 6};
  const std::uint8_t ref_values[] = {3, 9, 10, 5, 4, 7};

  image2d_view_host<std::uint8_t> img(data, 3, 2, 3, 1);
  auto                            d_img = amazonia::transfer(img);
  ASSERT_EQ(d_img.height(), 2);
  ASSERT_EQ(d_img.width(), 3);

  {
    dim3 grid_dim(1, 1);
    dim3 block_dim(3, 2);
    add_one<<<grid_dim, block_dim>>>(d_img);
    cudaDeviceSynchronize();
  }

  amazonia::transfer(d_img, img);

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      ASSERT_EQ(img(x, y), ref_values[y * 3 + x]);
  }
}

TEST(image2d, u32_device)
{
  using namespace amazonia;
  constexpr int e_size = sizeof(std::uint32_t);

  std::uint32_t       data[]       = {2, 8, 9, 4, 3, 6};
  const std::uint32_t ref_values[] = {3, 9, 10, 5, 4, 7};

  image2d_view_host<std::uint32_t> img(data, 3, 2, 3 * e_size, e_size);
  auto                             d_img = amazonia::transfer(img);
  ASSERT_EQ(d_img.height(), 2);
  ASSERT_EQ(d_img.width(), 3);
  {
    dim3 grid_dim(1, 1);
    dim3 block_dim(3, 2);
    add_one<<<grid_dim, block_dim>>>(d_img);
    cudaDeviceSynchronize();
  }

  amazonia::transfer(d_img, img);

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      ASSERT_EQ(img(x, y), ref_values[y * 3 + x]);
  }
}