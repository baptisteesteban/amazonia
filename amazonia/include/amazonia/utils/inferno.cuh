#pragma once

#include <amazonia/core/image2d.cuh>
#include <amazonia/core/rgb.cuh>

#include <algorithm>

namespace amazonia::utils
{
  /// \brief Colorize an univariate image using inferno colormap.
  /// \param in The input image.
  /// \param out The output image
  template <typename T>
  void colorize_inferno(const image2d_view_host<T>& in, image2d_view_host<rgb8>& out) noexcept;

  /// \brief Colorize an univariate image using inferno colormap.
  /// \param in The input image.
  /// \return The colorized image
  template <typename T>
  image2d_host<rgb8> colorize_inferno(const image2d_view_host<T>& in);
} // namespace amazonia::utils

/*
 * Implementations
 */

static constexpr amazonia::rgb8 inferno_mapping[] = {
    amazonia::rgb8{0, 0, 3},       //
    amazonia::rgb8{0, 0, 4},       //
    amazonia::rgb8{0, 0, 6},       //
    amazonia::rgb8{1, 0, 7},       //
    amazonia::rgb8{1, 1, 9},       //
    amazonia::rgb8{1, 1, 11},      //
    amazonia::rgb8{2, 1, 14},      //
    amazonia::rgb8{2, 2, 16},      //
    amazonia::rgb8{3, 2, 18},      //
    amazonia::rgb8{4, 3, 20},      //
    amazonia::rgb8{4, 3, 22},      //
    amazonia::rgb8{5, 4, 24},      //
    amazonia::rgb8{6, 4, 27},      //
    amazonia::rgb8{7, 5, 29},      //
    amazonia::rgb8{8, 6, 31},      //
    amazonia::rgb8{9, 6, 33},      //
    amazonia::rgb8{10, 7, 35},     //
    amazonia::rgb8{11, 7, 38},     //
    amazonia::rgb8{13, 8, 40},     //
    amazonia::rgb8{14, 8, 42},     //
    amazonia::rgb8{15, 9, 45},     //
    amazonia::rgb8{16, 9, 47},     //
    amazonia::rgb8{18, 10, 50},    //
    amazonia::rgb8{19, 10, 52},    //
    amazonia::rgb8{20, 11, 54},    //
    amazonia::rgb8{22, 11, 57},    //
    amazonia::rgb8{23, 11, 59},    //
    amazonia::rgb8{25, 11, 62},    //
    amazonia::rgb8{26, 11, 64},    //
    amazonia::rgb8{28, 12, 67},    //
    amazonia::rgb8{29, 12, 69},    //
    amazonia::rgb8{31, 12, 71},    //
    amazonia::rgb8{32, 12, 74},    //
    amazonia::rgb8{34, 11, 76},    //
    amazonia::rgb8{36, 11, 78},    //
    amazonia::rgb8{38, 11, 80},    //
    amazonia::rgb8{39, 11, 82},    //
    amazonia::rgb8{41, 11, 84},    //
    amazonia::rgb8{43, 10, 86},    //
    amazonia::rgb8{45, 10, 88},    //
    amazonia::rgb8{46, 10, 90},    //
    amazonia::rgb8{48, 10, 92},    //
    amazonia::rgb8{50, 9, 93},     //
    amazonia::rgb8{52, 9, 95},     //
    amazonia::rgb8{53, 9, 96},     //
    amazonia::rgb8{55, 9, 97},     //
    amazonia::rgb8{57, 9, 98},     //
    amazonia::rgb8{59, 9, 100},    //
    amazonia::rgb8{60, 9, 101},    //
    amazonia::rgb8{62, 9, 102},    //
    amazonia::rgb8{64, 9, 102},    //
    amazonia::rgb8{65, 9, 103},    //
    amazonia::rgb8{67, 10, 104},   //
    amazonia::rgb8{69, 10, 105},   //
    amazonia::rgb8{70, 10, 105},   //
    amazonia::rgb8{72, 11, 106},   //
    amazonia::rgb8{74, 11, 106},   //
    amazonia::rgb8{75, 12, 107},   //
    amazonia::rgb8{77, 12, 107},   //
    amazonia::rgb8{79, 13, 108},   //
    amazonia::rgb8{80, 13, 108},   //
    amazonia::rgb8{82, 14, 108},   //
    amazonia::rgb8{83, 14, 109},   //
    amazonia::rgb8{85, 15, 109},   //
    amazonia::rgb8{87, 15, 109},   //
    amazonia::rgb8{88, 16, 109},   //
    amazonia::rgb8{90, 17, 109},   //
    amazonia::rgb8{91, 17, 110},   //
    amazonia::rgb8{93, 18, 110},   //
    amazonia::rgb8{95, 18, 110},   //
    amazonia::rgb8{96, 19, 110},   //
    amazonia::rgb8{98, 20, 110},   //
    amazonia::rgb8{99, 20, 110},   //
    amazonia::rgb8{101, 21, 110},  //
    amazonia::rgb8{102, 21, 110},  //
    amazonia::rgb8{104, 22, 110},  //
    amazonia::rgb8{106, 23, 110},  //
    amazonia::rgb8{107, 23, 110},  //
    amazonia::rgb8{109, 24, 110},  //
    amazonia::rgb8{110, 24, 110},  //
    amazonia::rgb8{112, 25, 110},  //
    amazonia::rgb8{114, 25, 109},  //
    amazonia::rgb8{115, 26, 109},  //
    amazonia::rgb8{117, 27, 109},  //
    amazonia::rgb8{118, 27, 109},  //
    amazonia::rgb8{120, 28, 109},  //
    amazonia::rgb8{122, 28, 109},  //
    amazonia::rgb8{123, 29, 108},  //
    amazonia::rgb8{125, 29, 108},  //
    amazonia::rgb8{126, 30, 108},  //
    amazonia::rgb8{128, 31, 107},  //
    amazonia::rgb8{129, 31, 107},  //
    amazonia::rgb8{131, 32, 107},  //
    amazonia::rgb8{133, 32, 106},  //
    amazonia::rgb8{134, 33, 106},  //
    amazonia::rgb8{136, 33, 106},  //
    amazonia::rgb8{137, 34, 105},  //
    amazonia::rgb8{139, 34, 105},  //
    amazonia::rgb8{141, 35, 105},  //
    amazonia::rgb8{142, 36, 104},  //
    amazonia::rgb8{144, 36, 104},  //
    amazonia::rgb8{145, 37, 103},  //
    amazonia::rgb8{147, 37, 103},  //
    amazonia::rgb8{149, 38, 102},  //
    amazonia::rgb8{150, 38, 102},  //
    amazonia::rgb8{152, 39, 101},  //
    amazonia::rgb8{153, 40, 100},  //
    amazonia::rgb8{155, 40, 100},  //
    amazonia::rgb8{156, 41, 99},   //
    amazonia::rgb8{158, 41, 99},   //
    amazonia::rgb8{160, 42, 98},   //
    amazonia::rgb8{161, 43, 97},   //
    amazonia::rgb8{163, 43, 97},   //
    amazonia::rgb8{164, 44, 96},   //
    amazonia::rgb8{166, 44, 95},   //
    amazonia::rgb8{167, 45, 95},   //
    amazonia::rgb8{169, 46, 94},   //
    amazonia::rgb8{171, 46, 93},   //
    amazonia::rgb8{172, 47, 92},   //
    amazonia::rgb8{174, 48, 91},   //
    amazonia::rgb8{175, 49, 91},   //
    amazonia::rgb8{177, 49, 90},   //
    amazonia::rgb8{178, 50, 89},   //
    amazonia::rgb8{180, 51, 88},   //
    amazonia::rgb8{181, 51, 87},   //
    amazonia::rgb8{183, 52, 86},   //
    amazonia::rgb8{184, 53, 86},   //
    amazonia::rgb8{186, 54, 85},   //
    amazonia::rgb8{187, 55, 84},   //
    amazonia::rgb8{189, 55, 83},   //
    amazonia::rgb8{190, 56, 82},   //
    amazonia::rgb8{191, 57, 81},   //
    amazonia::rgb8{193, 58, 80},   //
    amazonia::rgb8{194, 59, 79},   //
    amazonia::rgb8{196, 60, 78},   //
    amazonia::rgb8{197, 61, 77},   //
    amazonia::rgb8{199, 62, 76},   //
    amazonia::rgb8{200, 62, 75},   //
    amazonia::rgb8{201, 63, 74},   //
    amazonia::rgb8{203, 64, 73},   //
    amazonia::rgb8{204, 65, 72},   //
    amazonia::rgb8{205, 66, 71},   //
    amazonia::rgb8{207, 68, 70},   //
    amazonia::rgb8{208, 69, 68},   //
    amazonia::rgb8{209, 70, 67},   //
    amazonia::rgb8{210, 71, 66},   //
    amazonia::rgb8{212, 72, 65},   //
    amazonia::rgb8{213, 73, 64},   //
    amazonia::rgb8{214, 74, 63},   //
    amazonia::rgb8{215, 75, 62},   //
    amazonia::rgb8{217, 77, 61},   //
    amazonia::rgb8{218, 78, 59},   //
    amazonia::rgb8{219, 79, 58},   //
    amazonia::rgb8{220, 80, 57},   //
    amazonia::rgb8{221, 82, 56},   //
    amazonia::rgb8{222, 83, 55},   //
    amazonia::rgb8{223, 84, 54},   //
    amazonia::rgb8{224, 86, 52},   //
    amazonia::rgb8{226, 87, 51},   //
    amazonia::rgb8{227, 88, 50},   //
    amazonia::rgb8{228, 90, 49},   //
    amazonia::rgb8{229, 91, 48},   //
    amazonia::rgb8{230, 92, 46},   //
    amazonia::rgb8{230, 94, 45},   //
    amazonia::rgb8{231, 95, 44},   //
    amazonia::rgb8{232, 97, 43},   //
    amazonia::rgb8{233, 98, 42},   //
    amazonia::rgb8{234, 100, 40},  //
    amazonia::rgb8{235, 101, 39},  //
    amazonia::rgb8{236, 103, 38},  //
    amazonia::rgb8{237, 104, 37},  //
    amazonia::rgb8{237, 106, 35},  //
    amazonia::rgb8{238, 108, 34},  //
    amazonia::rgb8{239, 109, 33},  //
    amazonia::rgb8{240, 111, 31},  //
    amazonia::rgb8{240, 112, 30},  //
    amazonia::rgb8{241, 114, 29},  //
    amazonia::rgb8{242, 116, 28},  //
    amazonia::rgb8{242, 117, 26},  //
    amazonia::rgb8{243, 119, 25},  //
    amazonia::rgb8{243, 121, 24},  //
    amazonia::rgb8{244, 122, 22},  //
    amazonia::rgb8{245, 124, 21},  //
    amazonia::rgb8{245, 126, 20},  //
    amazonia::rgb8{246, 128, 18},  //
    amazonia::rgb8{246, 129, 17},  //
    amazonia::rgb8{247, 131, 16},  //
    amazonia::rgb8{247, 133, 14},  //
    amazonia::rgb8{248, 135, 13},  //
    amazonia::rgb8{248, 136, 12},  //
    amazonia::rgb8{248, 138, 11},  //
    amazonia::rgb8{249, 140, 9},   //
    amazonia::rgb8{249, 142, 8},   //
    amazonia::rgb8{249, 144, 8},   //
    amazonia::rgb8{250, 145, 7},   //
    amazonia::rgb8{250, 147, 6},   //
    amazonia::rgb8{250, 149, 6},   //
    amazonia::rgb8{250, 151, 6},   //
    amazonia::rgb8{251, 153, 6},   //
    amazonia::rgb8{251, 155, 6},   //
    amazonia::rgb8{251, 157, 6},   //
    amazonia::rgb8{251, 158, 7},   //
    amazonia::rgb8{251, 160, 7},   //
    amazonia::rgb8{251, 162, 8},   //
    amazonia::rgb8{251, 164, 10},  //
    amazonia::rgb8{251, 166, 11},  //
    amazonia::rgb8{251, 168, 13},  //
    amazonia::rgb8{251, 170, 14},  //
    amazonia::rgb8{251, 172, 16},  //
    amazonia::rgb8{251, 174, 18},  //
    amazonia::rgb8{251, 176, 20},  //
    amazonia::rgb8{251, 177, 22},  //
    amazonia::rgb8{251, 179, 24},  //
    amazonia::rgb8{251, 181, 26},  //
    amazonia::rgb8{251, 183, 28},  //
    amazonia::rgb8{251, 185, 30},  //
    amazonia::rgb8{250, 187, 33},  //
    amazonia::rgb8{250, 189, 35},  //
    amazonia::rgb8{250, 191, 37},  //
    amazonia::rgb8{250, 193, 40},  //
    amazonia::rgb8{249, 195, 42},  //
    amazonia::rgb8{249, 197, 44},  //
    amazonia::rgb8{249, 199, 47},  //
    amazonia::rgb8{248, 201, 49},  //
    amazonia::rgb8{248, 203, 52},  //
    amazonia::rgb8{248, 205, 55},  //
    amazonia::rgb8{247, 207, 58},  //
    amazonia::rgb8{247, 209, 60},  //
    amazonia::rgb8{246, 211, 63},  //
    amazonia::rgb8{246, 213, 66},  //
    amazonia::rgb8{245, 215, 69},  //
    amazonia::rgb8{245, 217, 72},  //
    amazonia::rgb8{244, 219, 75},  //
    amazonia::rgb8{244, 220, 79},  //
    amazonia::rgb8{243, 222, 82},  //
    amazonia::rgb8{243, 224, 86},  //
    amazonia::rgb8{243, 226, 89},  //
    amazonia::rgb8{242, 228, 93},  //
    amazonia::rgb8{242, 230, 96},  //
    amazonia::rgb8{241, 232, 100}, //
    amazonia::rgb8{241, 233, 104}, //
    amazonia::rgb8{241, 235, 108}, //
    amazonia::rgb8{241, 237, 112}, //
    amazonia::rgb8{241, 238, 116}, //
    amazonia::rgb8{241, 240, 121}, //
    amazonia::rgb8{241, 242, 125}, //
    amazonia::rgb8{242, 243, 129}, //
    amazonia::rgb8{242, 244, 133}, //
    amazonia::rgb8{243, 246, 137}, //
    amazonia::rgb8{244, 247, 141}, //
    amazonia::rgb8{245, 248, 145}, //
    amazonia::rgb8{246, 250, 149}, //
    amazonia::rgb8{247, 251, 153}, //
    amazonia::rgb8{249, 252, 157}, //
    amazonia::rgb8{250, 253, 160}, //
    amazonia::rgb8{252, 254, 164}  //
};

namespace amazonia::utils
{
  template <typename T>
  std::pair<T, T> minmax(const image2d_view_host<T>& in) noexcept
  {
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    for (int l = 0; l < in.nrows(); l++)
    {
      for (int c = 0; c < in.ncols(); c++)
      {
        min = std::min(min, in(l, c));
        max = std::max(max, in(l, c));
      }
    }
    return {min, max};
  }

  template <typename T>
  void colorize_inferno(const image2d_view_host<T>& in, image2d_view_host<rgb8>& out) noexcept
  {
    assert(in.nrows() == out.nrows() && in.ncols() == out.ncols());
    const auto [min_v, max_v] = minmax(in);
    for (int l = 0; l < out.nrows(); l++)
    {
      for (int c = 0; c < out.ncols(); c++)
      {
        const int ind = (static_cast<float>(in(l, c)) - min_v) / (max_v - min_v) * 255;
        out(l, c)     = inferno_mapping[ind];
      }
    }
  }

  template <typename T>
  image2d_host<rgb8> colorize_inferno(const image2d_view_host<T>& in)
  {
    image2d_host<rgb8> out(in.nrows(), in.ncols());
    colorize_inferno(in, out);
    return out;
  }
} // namespace amazonia::utils