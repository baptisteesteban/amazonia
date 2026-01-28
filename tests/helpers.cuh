#pragma once

#include <amazonia/core/image2d_view.cuh>
#include <amazonia/core/rgb.cuh>

#include <gtest/gtest.h>

#include <concepts>
#include <format>

namespace amazonia::tests
{
  template <typename T>
  testing::AssertionResult images_equality_comparison(const image2d_view_host<T>& img1,
                                                      const image2d_view_host<T>& img2)
  {
    if (img1.width() != img2.width())
      return testing::AssertionFailure() << std::format("The images have different width (img1: {}, img2: {})",
                                                        img1.width(), img2.width());
    if (img1.height() != img2.height())
      return testing::AssertionFailure() << std::format("The images have different height (img1: {}, img2: {})",
                                                        img1.height(), img2.height());

    for (int y = 0; y < img1.height(); y++)
    {
      for (int x = 0; x < img1.width(); x++)
      {
        if (img1(x, y) != img2(x, y))
        {
          if constexpr (std::same_as<T, rgb8>)
          {
            return testing::AssertionFailure()
                   << std::format("img1({}, {}) ({}, {}, {}) != img2({}, {}) ({}, {}, {})", x, y, img1(x, y).r,
                                  img1(x, y).g, img1(x, y).b, x, y, img2(x, y).r, img2(x, y).g, img2(x, y).b);
          }
          else
          {
            return testing::AssertionFailure()
                   << std::format("img1({}, {}) ({}) != img2({}, {}) ({})", x, y, img1(x, y), x, y, img2(x, y));
          }
        }
      }
    }
    return testing::AssertionSuccess();
  }
} // namespace amazonia::tests

#define ASSERT_IMAGES_EQ(img1, img2) ASSERT_TRUE(amazonia::tests::images_equality_comparison(img1, img2));