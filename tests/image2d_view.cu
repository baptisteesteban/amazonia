#include <amazonia/core/image2d_view.cuh>

#include <gtest/gtest.h>

TEST(image2d_view, u8)
{
  using namespace amazonia;

  std::uint8_t data[]    = {1, 3, 9, 7, 6, 2};
  int          shapes[]  = {2, 3};
  int          strides[] = {3, 1};

  image2d_view_host<std::uint8_t> view(data, shapes, strides);
  ASSERT_EQ(view.nrows(), 2);
  ASSERT_EQ(view.ncols(), 3);
  ASSERT_EQ(view.shape(0), view.nrows());
  ASSERT_EQ(view.shape(1), view.ncols());
  ASSERT_EQ(view.stride(0), 3);
  ASSERT_EQ(view.stride(1), 1);

  for (int l = 0; l < view.nrows(); l++)
  {
    for (int c = 0; c < view.ncols(); c++)
      ASSERT_EQ(view(l, c), data[l * 3 + c]);
  }
}

TEST(image2d_view, u32)
{
  using namespace amazonia;
  using data_type_t    = std::uint32_t;
  constexpr int e_size = sizeof(data_type_t);

  data_type_t data[]    = {1, 3, 9, 7, 6, 2};
  int         shapes[]  = {2, 3};
  int         strides[] = {3 * e_size, e_size};

  image2d_view_host<data_type_t> view(data, shapes, strides);
  ASSERT_EQ(view.nrows(), 2);
  ASSERT_EQ(view.ncols(), 3);
  ASSERT_EQ(view.shape(0), view.nrows());
  ASSERT_EQ(view.shape(1), view.ncols());
  ASSERT_EQ(view.stride(0), 3 * e_size);
  ASSERT_EQ(view.stride(1), e_size);

  for (int l = 0; l < view.nrows(); l++)
  {
    for (int c = 0; c < view.ncols(); c++)
      ASSERT_EQ(view(l, c), data[l * 3 + c]);
  }
}