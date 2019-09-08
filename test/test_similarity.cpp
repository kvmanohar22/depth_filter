#include "depth_filter/similarity.h"
#include "depth_filter/global.h"
 
using namespace std;
 
void test1() {
  uint8_t ref_patch1_[] = {1, 2, 3, 4};
  uint8_t cur_patch1_[] = {1, 2, 3, 4};
 
  static constexpr size_t half_patch_size = 2;
  uint8_t ref_patch_[] = {76, 72, 82, 77, 68, 68, 79, 90, 77, 81, 84, 71, 72, 70, 72, 70};
  uint8_t cur_patch_[] = {76, 72, 82, 77, 68, 68, 79, 90, 77, 81, 84, 71, 72, 70, 72, 70};
  
  utils::similarity::ZMSSD<uint8_t, half_patch_size> ssd_uint8_t(ref_patch_);
  cout << "ssd similarity = " << ssd_uint8_t.similarity(cur_patch_, 2*half_patch_size) << endl;

  utils::similarity::ZMSSD<uint8_t, half_patch_size> zmssd_uint8_t(ref_patch_);
  cout << "zmssd similarity = " << zmssd_uint8_t.similarity(cur_patch_, 2*half_patch_size) << endl;

  utils::similarity::NCC<uint8_t, half_patch_size> ncc_uint8_t(ref_patch_);
  cout << "ncc similarity = " << ncc_uint8_t.similarity(cur_patch_, 2*half_patch_size) << endl;

  utils::similarity::ZMNCC<uint8_t, half_patch_size> zmncc_uint8_t(ref_patch_);
  cout << "zmncc similarity = " << zmncc_uint8_t.similarity(cur_patch_, 2*half_patch_size) << endl;
}

void test2() {
  static constexpr size_t half_patch_size = 4;
  size_t patch_size = 2*half_patch_size;
  size_t start_idx_x = 260;
  size_t start_idx_y = 270;
 
  string base = std::getenv("PROJECT_DF");
  string imgfile = base + "/data/test_image.png";
  cv::Mat img = cv::imread(imgfile.c_str(), 0);
  assert(!img.empty());
  cv::imshow("img", img);
  cv::waitKey(0);

  // create reference patch
  uint8_t* img_ptr = img.ptr<uint8_t>();
  uint8_t  ref_patch[patch_size*patch_size] __attribute__ ((aligned(16)));
  uint8_t  cur_patch[patch_size*patch_size] __attribute__ ((aligned(16)));
  for(size_t y=0;y<patch_size;++y)
    for(size_t x=0;x<patch_size;++x)
      *(ref_patch+y*patch_size+x) = *(img_ptr+(start_idx_y+y)*img.cols+start_idx_x+x);  

  std::ofstream ofile(base+"/data/zmssd_similarity.txt");
  utils::similarity::ZMSSD<uint8_t, half_patch_size> zmssd(ref_patch);
  for(size_t x=0; x<img.cols; x+=half_patch_size) {
    uint8_t* cur_patch_ptr = img_ptr+start_idx_y*img.cols+x;
    double similarity = zmssd.similarity(cur_patch_ptr, img.cols);
    ofile << x << " "  << similarity << endl; 
  }
  ofile.close();

  ofile.open(base+"/data/zmncc_similarity.txt");
  utils::similarity::NCC<uint8_t, half_patch_size> zmncc(ref_patch);
  for(size_t x=0; x<img.cols; x+=half_patch_size) {
    uint8_t* cur_patch_ptr = img_ptr+start_idx_y*img.cols+x;
    double similarity = zmncc.similarity(cur_patch_ptr, img.cols);
    ofile << x << " "  << similarity << endl;
  }

}

void test3() {
  static constexpr size_t half_patch_size = 4;
  size_t patch_size = 2*half_patch_size;
  size_t start_idx_x = 272;
  size_t start_idx_y = 348;
 
  string base = std::getenv("PROJECT_DF");
  string imgfile = base + "/data/test_image.png";
  cv::Mat img = cv::imread(imgfile.c_str(), 0);
  assert(!img.empty());
  cv::imshow("img", img);
  cv::waitKey(0);

  // create reference patch
  uint8_t* img_ptr = img.ptr<uint8_t>();
  uint8_t  ref_patch[patch_size*patch_size] __attribute__ ((aligned(16)));
  uint8_t  cur_patch[patch_size*patch_size] __attribute__ ((aligned(16)));
  for(size_t y=0;y<patch_size;++y)
    for(size_t x=0;x<patch_size;++x)
      *(ref_patch+y*patch_size+x) = *(img_ptr+(start_idx_y+y)*img.cols+start_idx_x+x);  

  utils::similarity::ZMNCC<uint8_t, half_patch_size> zmncc(ref_patch);

  // ref patch 1
  std::ofstream ofile;
  ofile.open(base+"/data/zmncc_test3_similarity.txt");
  for(size_t x=0; x<img.cols; x+=half_patch_size) {
    uint8_t* cur_patch_ptr = img_ptr+start_idx_y*img.cols+x;
    double similarity = zmncc.similarity(cur_patch_ptr, img.cols);
    ofile << x << " "  << similarity << endl;
  }
  ofile.close();

  // ref patch 2
  start_idx_x = 260;
  start_idx_y = 270;
  for(size_t y=0;y<patch_size;++y)
    for(size_t x=0;x<patch_size;++x)
      *(ref_patch+y*patch_size+x) = *(img_ptr+(start_idx_y+y)*img.cols+start_idx_x+x);  
  zmncc.set_ref_patch(ref_patch);
  ofile.open(base+"/data/zmncc_test3_similarity2.txt");
  for(size_t x=0; x<img.cols; x+=half_patch_size) {
    uint8_t* cur_patch_ptr = img_ptr+start_idx_y*img.cols+x;
    double similarity = zmncc.similarity(cur_patch_ptr, img.cols);
    ofile << x << " "  << similarity << endl;
  }
  ofile.close();

  // ref patch 1
  start_idx_x = 272;
  start_idx_y = 348;
  for(size_t y=0;y<patch_size;++y)
    for(size_t x=0;x<patch_size;++x)
      *(ref_patch+y*patch_size+x) = *(img_ptr+(start_idx_y+y)*img.cols+start_idx_x+x);  
  zmncc.set_ref_patch(ref_patch);
  ofile.open(base+"/data/zmncc_test3_similarity3.txt");
  for(size_t x=0; x<img.cols; x+=half_patch_size) {
    uint8_t* cur_patch_ptr = img_ptr+start_idx_y*img.cols+x;
    double similarity = zmncc.similarity(cur_patch_ptr, img.cols);
    ofile << x << " "  << similarity << endl;
  }
}

void test4() {
  static constexpr size_t half_patch_size = 4;
  size_t patch_size = 2*half_patch_size;
  size_t start_idx_x = 272;
  size_t start_idx_y = 348;
 
  string base = std::getenv("PROJECT_DF");
  string imgfile = base + "/data/test_image.png";
  cv::Mat img = cv::imread(imgfile.c_str(), 0);
  assert(!img.empty());
  cv::imshow("img", img);
  cv::waitKey(0);

  // create reference patch
  uint8_t* img_ptr = img.ptr<uint8_t>();
  uint8_t  ref_patch[patch_size*patch_size] __attribute__ ((aligned(16)));
  uint8_t  cur_patch[patch_size*patch_size] __attribute__ ((aligned(16)));
  for(size_t y=0;y<patch_size;++y)
    for(size_t x=0;x<patch_size;++x)
      *(ref_patch+y*patch_size+x) = *(img_ptr+(start_idx_y+y)*img.cols+start_idx_x+x);  

  utils::similarity::ZMNCC<uint8_t, half_patch_size> zmncc;
  zmncc.set_ref_patch(ref_patch);

  // ref patch 1
  std::ofstream ofile;
  ofile.open(base+"/data/zmncc_test4_similarity.txt");
  for(size_t x=0; x<img.cols; x+=half_patch_size) {
    uint8_t* cur_patch_ptr = img_ptr+start_idx_y*img.cols+x;
    double similarity = zmncc.similarity(cur_patch_ptr, img.cols);
    ofile << x << " "  << similarity << endl;
  }
  ofile.close();
}

int main() {
  test1();
  test2();
  test3();
  test4();
}
