#pragma once

#include <cstdint>
#include <format>

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

  /// \brief Equality operator for `rgb` data structure.
  /// \param lhs The left operand
  /// \param rhs The right operand
  /// \return Are the two value equals ?
  template <typename T>
  __host__ __device__ bool operator==(const rgb<T>& lhs, const rgb<T>& rhs) noexcept;

  /// \brief Inequality operator for `rgb` data structure.
  /// \param lhs The left operand
  /// \param rhs The right operand
  /// \return Are the two value unequals ?
  template <typename T>
  __host__ __device__ bool operator!=(const rgb<T>& lhs, const rgb<T>& rhs) noexcept;

  /// \brief Alias for standard `rgb8` triplet.
  using rgb8 = rgb<std::uint8_t>;

  /*
   * Implementation
   */
  template <typename T>
  __host__ __device__ bool operator==(const rgb<T>& lhs, const rgb<T>& rhs) noexcept
  {
    return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b;
  }

  template <typename T>
  __host__ __device__ bool operator!=(const rgb<T>& lhs, const rgb<T>& rhs) noexcept
  {
    return !(lhs == rhs);
  }
} // namespace amazonia