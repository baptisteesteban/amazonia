#include <amazonia/core/image2d.cuh>
#include <amazonia/core/transfer.cuh>

#include <gtest/gtest.h>

TEST(image2d, u8_host)
{
  using namespace amazonia;

  std::uint8_t data[] = {2, 8, 9, 4, 3, 6};

  image2d_host<std::uint8_t> img(2, 3);
  ASSERT_EQ(img.nrows(), 2);
  ASSERT_EQ(img.ncols(), 3);
  ASSERT_EQ(img.shape(0), img.nrows());
  ASSERT_EQ(img.shape(1), img.ncols());
  ASSERT_EQ(img.stride(0), 3);
  ASSERT_EQ(img.stride(1), 1);

  for (int l = 0; l < img.nrows(); l++)
  {
    for (int c = 0; c < img.ncols(); c++)
      img(l, c) = data[l * 3 + c];
  }

  for (int l = 0; l < img.nrows(); l++)
  {
    for (int c = 0; c < img.ncols(); c++)
      ASSERT_EQ(img(l, c), data[l * 3 + c]);
  }
}

TEST(image2d, u32_host)
{
  using namespace amazonia;
  using value_t        = std::uint32_t;
  constexpr int e_size = sizeof(value_t);

  value_t data[] = {2, 8, 9, 4, 3, 6};

  image2d_host<value_t> img(2, 3);
  ASSERT_EQ(img.nrows(), 2);
  ASSERT_EQ(img.ncols(), 3);
  ASSERT_EQ(img.shape(0), img.nrows());
  ASSERT_EQ(img.shape(1), img.ncols());
  ASSERT_EQ(img.stride(0), 3 * e_size);
  ASSERT_EQ(img.stride(1), e_size);

  for (int l = 0; l < img.nrows(); l++)
  {
    for (int c = 0; c < img.ncols(); c++)
      img(l, c) = data[l * 3 + c];
  }

  for (int l = 0; l < img.nrows(); l++)
  {
    for (int c = 0; c < img.ncols(); c++)
      ASSERT_EQ(img(l, c), data[l * 3 + c]);
  }
}

template <typename T>
__global__ void add_one(amazonia::image2d_view<T, amazonia::device_t>& img)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < img.ncols() && y < img.nrows())
    img(y, x) += 1;
}

TEST(image2d, u8_device)
{
  using namespace amazonia;

  std::uint8_t       data[]       = {2, 8, 9, 4, 3, 6};
  const std::uint8_t ref_values[] = {3, 9, 10, 5, 4, 7};
  int                shapes[]     = {2, 3};
  int                strides[]    = {3, 1};

  image2d_view_host<std::uint8_t> img(data, shapes, strides);
  auto                            d_img = amazonia::transfer(img);
  ASSERT_EQ(d_img.nrows(), 2);
  ASSERT_EQ(d_img.ncols(), 3);
  ASSERT_EQ(d_img.shape(0), d_img.nrows());
  ASSERT_EQ(d_img.shape(1), d_img.ncols());

  {
    dim3 grid_dim(1, 1);
    dim3 block_dim(3, 2);
    add_one<<<grid_dim, block_dim>>>(d_img);
    cudaDeviceSynchronize();
  }

  amazonia::transfer(d_img, img);

  for (int l = 0; l < img.nrows(); l++)
  {
    for (int c = 0; c < img.ncols(); c++)
      ASSERT_EQ(img(l, c), ref_values[l * 3 + c]);
  }
}

TEST(image2d, u32_device)
{
  using namespace amazonia;

  std::uint32_t       data[]       = {2, 8, 9, 4, 3, 6};
  const std::uint32_t ref_values[] = {3, 9, 10, 5, 4, 7};
  int                 shapes[]     = {2, 3};
  int                 strides[]    = {3 * sizeof(std::uint32_t), sizeof(std::uint32_t)};

  image2d_view_host<std::uint32_t> img(data, shapes, strides);
  auto                             d_img = amazonia::transfer(img);
  ASSERT_EQ(d_img.nrows(), 2);
  ASSERT_EQ(d_img.ncols(), 3);
  ASSERT_EQ(d_img.shape(0), d_img.nrows());
  ASSERT_EQ(d_img.shape(1), d_img.ncols());

  {
    dim3 grid_dim(1, 1);
    dim3 block_dim(3, 2);
    add_one<<<grid_dim, block_dim>>>(d_img);
    cudaDeviceSynchronize();
  }

  amazonia::transfer(d_img, img);

  for (int l = 0; l < img.nrows(); l++)
  {
    for (int c = 0; c < img.ncols(); c++)
      ASSERT_EQ(img(l, c), ref_values[l * 3 + c]);
  }
}