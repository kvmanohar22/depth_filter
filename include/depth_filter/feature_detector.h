#ifndef _DF_FEATURE_H_
#define _DF_FEATURE_H_

#include "depth_filter/global.h"
#include "depth_filter/corner.h"
#include "depth_filter/frame.h"

namespace df {

class FastDetector {
public:
 
  FastDetector() =default;
 ~FastDetector() =default;
  
  static void detect(FramePtr &frame, list<Corner*>& corners) {
    cv::Mat img = frame->img_.clone();
    vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect(img, keypoints);

    std::for_each(keypoints.begin(), keypoints.end(), [&](cv::KeyPoint &kpt) {
      Vector2d pt(kpt.pt.x, kpt.pt.y);
      Corner *corner = new df::Corner(pt, frame);
      corners.push_back(corner);
    });
  }
};

}

#endif

