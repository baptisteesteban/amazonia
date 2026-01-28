#include <amazonia/core/rgb.cuh>
#include <amazonia/io/imread.cuh>
#include <amazonia/io/imwrite.cuh>

#include <gtest/gtest.h>

#include "helpers.cuh"

TEST(IO, image2d_u8)
{
  using namespace amazonia;

  std::uint8_t data[] = {4, 7, 9, 7, 6, 5};
  auto         in     = image2d_view_host<std::uint8_t>(data, 3, 2, 3, 1);
  io::imwrite("tmp.png", in);

  image2d_host<std::uint8_t> loaded;
  io::imread("tmp.png", loaded);

  ASSERT_EQ(loaded.width(), in.width());
  ASSERT_EQ(loaded.height(), in.height());
  ASSERT_EQ(loaded.spitch(), in.spitch());
  ASSERT_EQ(loaded.epitch(), in.epitch());

  ASSERT_IMAGES_EQ(loaded, in);
}

TEST(IO, image2d_rgb8)
{
  using namespace amazonia;

  constexpr std::size_t e_size = sizeof(rgb8);

  rgb8 data[]    = {{7, 18, 245}, {75, 46, 91}, {13, 64, 85}, {125, 33, 64}, {48, 26, 75}, {0, 68, 7}};
  int  shapes[]  = {2, 3};
  int  strides[] = {3 * e_size, e_size};
  auto in        = image2d_view_host<rgb8>(data, 3, 2, 3 * e_size, e_size);
  io::imwrite("tmp.png", in);

  image2d_host<rgb8> loaded;
  io::imread("tmp.png", loaded);

  ASSERT_EQ(loaded.width(), in.width());
  ASSERT_EQ(loaded.height(), in.height());
  ASSERT_EQ(loaded.spitch(), in.spitch());
  ASSERT_EQ(loaded.epitch(), in.epitch());

  ASSERT_IMAGES_EQ(loaded, in);
}