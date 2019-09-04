#include "depth_filter/global.hpp"
#include "depth_filter/utils.h"
#include "depth_filter/viewer.h"

using namespace std;

namespace df {

void visualize() {
  string root = std::getenv("PROJECT_DF");
  vector<Vector3d> xyzs, colors;
  vector<Matrix4d> poses;

  string filename(root+"/analysis/forward/log1/depth_filter_rpg_synthetic_forward_points.ply");
  ifstream ifile(filename.c_str());
  assert(ifile.is_open());
  while(!ifile.eof()) {
    string s;
    std::getline(ifile, s);
    if(s.empty())
      continue;
    Vector3d xyz, color;
    std::istringstream ss(s);
    ss >> xyz(0);
    ss >> xyz(1);
    ss >> xyz(2);
    ss >> color(0);
    ss >> color(1);
    ss >> color(2);
    xyzs.push_back(xyz);
    colors.push_back(color);
  }
  ifile.close();

  string posefile(root+"/analysis/forward/log1/depth_filter_rpg_synthetic_forward_poses.txt");
  ifile.open(posefile.c_str());
  assert(ifile.is_open());
  while (!ifile.eof()) {
    double pose[7];
    std::string s;
    std::getline(ifile, s);
    std::istringstream ss(s);
    size_t num=0;
    if (s.empty())
      continue;
    double t;
    ss >> t;
    while (num < 7) {
      ss >> pose[num];
      ++num; 
    }
    Matrix3d R_w_f; Vector3d t_w_f;
    for (size_t i=0;i<3;++i)
      t_w_f(i) = pose[i];
    df::utils::quaternion_to_rotation_matrix(pose+3, R_w_f);
    Sophus::SE3 T_w_f(R_w_f, t_w_f);
    poses.emplace_back(T_w_f.matrix());
  }
  cout << "#poses = " << poses.size() << endl
       << "#points = " << xyzs.size() << endl;

  df::Viewer3D viewer(xyzs, colors, poses);
  while(true);
}

}

int main(int argc, char** argv) {
  df::visualize();
}
