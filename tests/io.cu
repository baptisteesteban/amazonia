#include <amazonia/io/imread.cuh>
#include <amazonia/io/imwrite.cuh>

#include <gtest/gtest.h>

TEST(IO, image2d_u8)
{
  using namespace amazonia;

  std::uint8_t data[]    = {4, 7, 9, 7, 6, 5};
  int          shapes[]  = {2, 3};
  int          strides[] = {3, 1};
  auto         in        = image2d_view_host<std::uint8_t>(data, shapes, strides);
  io::imwrite("tmp.png", in);

  image2d_host<std::uint8_t> loaded;
  io::imread("tmp.png", loaded);

  for (int i = 0; i < 2; i++)
  {
    ASSERT_EQ(loaded.shape(i), in.shape(i));
    ASSERT_EQ(loaded.stride(i), in.stride(i));
  }
  for (int l = 0; l < loaded.nrows(); l++)
  {
    for (int c = 0; c < loaded.ncols(); c++)
      ASSERT_EQ(loaded(l, c), in(l, c));
  }
}