#include <amazonia/core/image2d.cuh>

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