#include <amazonia/core/image2d.cuh>
#include <amazonia/core/transfer.cuh>
#include <amazonia/dt/generalised_geodesic.cuh>
#include <amazonia/io/imread.cuh>
#include <amazonia/io/imwrite.cuh>
#include <amazonia/utils/inferno.cuh>

#include <iostream>

int main(int argc, char* argv[])
{
  if (argc < 5)
  {
    std::cerr << "Usage: " << argv[0] << " input_image input_mask lambda output_colorized\n";
    return 1;
  }
  amazonia::image2d_host<std::uint8_t> img;
  amazonia::io::imread(argv[1], img);
  amazonia::image2d_host<std::uint8_t> mask;
  amazonia::io::imread(argv[2], mask);
  float lambda = std::atof(argv[3]);

  const auto d_img     = amazonia::transfer(img);
  const auto d_mask    = amazonia::transfer(mask);
  auto       d_dist    = amazonia::dt::generalised_geodesic(d_img, d_mask, lambda);
  const auto dist      = amazonia::transfer(d_dist);
  const auto colorized = amazonia::utils::colorize_inferno(dist);

  amazonia::io::imwrite(argv[4], colorized);

  return 0;
}