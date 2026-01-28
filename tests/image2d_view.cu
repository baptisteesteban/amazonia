#include <amazonia/core/image2d_view.cuh>

#include <gtest/gtest.h>

TEST(image2d_view, u8)
{
  using namespace amazonia;

  std::uint8_t data[] = {1, 3, 9, 7, 6, 2};

  image2d_view_host<std::uint8_t> view(data, 3, 2, 3, 1);
  ASSERT_EQ(view.height(), 2);
  ASSERT_EQ(view.width(), 3);
  ASSERT_EQ(view.spitch(), 3);
  ASSERT_EQ(view.epitch(), 1);

  for (int y = 0; y < view.height(); y++)
  {
    for (int x = 0; x < view.width(); x++)
      ASSERT_EQ(view(x, y), data[y * 3 + x]);
  }
}

TEST(image2d_view, u32)
{
  using namespace amazonia;
  using data_type_t    = std::uint32_t;
  constexpr int e_size = sizeof(data_type_t);

  data_type_t data[] = {1, 3, 9, 7, 6, 2};

  image2d_view_host<data_type_t> view(data, 3, 2, 3 * e_size, e_size);
  ASSERT_EQ(view.height(), 2);
  ASSERT_EQ(view.width(), 3);
  ASSERT_EQ(view.spitch(), 3 * e_size);
  ASSERT_EQ(view.epitch(), e_size);

  for (int y = 0; y < view.height(); y++)
  {
    for (int x = 0; x < view.width(); x++)
      ASSERT_EQ(view(x, y), data[y * 3 + x]);
  }
}