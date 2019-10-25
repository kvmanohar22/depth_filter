#include "depth_filter/global.h"
#include "depth_filter/utils.h"
#include "depth_filter/viewer.h"

using namespace std;

namespace df {

void visualize() {
  string root = std::getenv("PROJECT_DF");
  vector<Vector3d> xyzs, colors;
  vector<Matrix4d> poses;

  // // Downward facing camera
  string filename(root+"/analysis/downward/log3/groundtruth_points_frame_2.txt");
  // string filename(root+"/analysis/downward/log3/depth_filter_sin2_tex2_h1_v8_d_points.txt");

  // // Forward facing camera
  // string filename(root+"/analysis/forward/log5/groundtruth_points_frame_600.txt");
  // string filename(root+"/analysis/forward/log8/depth_filter_rpg_synthetic_forward_points.txt");

  ifstream ifile(filename.c_str());
  assert(ifile.is_open());
  while(!ifile.eof()) {
    string s;
    std::getline(ifile, s);
    if(s.empty())
      continue;
    Vector3d xyz, color(255,255,255);
    std::istringstream ss(s);
    ss >> xyz(0);
    ss >> xyz(1);
    ss >> xyz(2);
    xyzs.push_back(xyz);
    colors.push_back(color);
  }
  ifile.close();
  cout << "#points = " << xyzs.size() << endl;

  df::Viewer3D viewer(xyzs, colors, poses);
  while(true);
}

}

int main(int argc, char** argv) {
  df::visualize();
}
