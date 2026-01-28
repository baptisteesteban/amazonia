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
    if (img1.nrows() != img2.nrows())
      return testing::AssertionFailure() << std::format("The images have different number of rows (img1: {}, img2: {})",
                                                        img1.nrows(), img2.nrows());
    if (img1.ncols() != img2.ncols())
      return testing::AssertionFailure() << std::format(
                 "The images have different number of columns (img1: {}, img2: {})", img1.ncols(), img2.ncols());

    for (int l = 0; l < img1.nrows(); l++)
    {
      for (int c = 0; c < img1.ncols(); c++)
      {
        if (img1(l, c) != img2(l, c))
        {
          if constexpr (std::same_as<T, rgb8>)
          {
            return testing::AssertionFailure()
                   << std::format("img1({}, {}) ({}, {}, {}) != img2({}, {}) ({}, {}, {})", l, c, img1(l, c).r,
                                  img1(l, c).g, img1(l, c).b, l, c, img2(l, c).r, img2(l, c).g, img2(l, c).b);
          }
          else
          {
            return testing::AssertionFailure()
                   << std::format("img1({}, {}) ({}) != img2({}, {}) ({})", l, c, img1(l, c), l, c, img2(l, c));
          }
        }
      }
    }
    return testing::AssertionSuccess();
  }
} // namespace amazonia::tests

#define ASSERT_IMAGES_EQ(img1, img2) ASSERT_TRUE(amazonia::tests::images_equality_comparison(img1, img2));