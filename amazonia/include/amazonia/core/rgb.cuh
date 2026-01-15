#pragma once

#include <cstdint>

namespace amazonia
{
  /// \brief Data structure representing a (r, g, b) triplet to represent
  /// a color
  /// \tparam T The data type of a color component
  template <typename T>
  struct rgb
  {
    T r; ///< The red component
    T g; ///< The green component
    T b; ///< The blue component
  };

  using rgb8 = rgb<std::uint8_t>;
} // namespace amazonia