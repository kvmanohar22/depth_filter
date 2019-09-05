#include "depth_filter/viewer.h"

namespace df {

Viewer3D::Viewer3D(
  vector<Vector3d>& points,
  vector<Vector3d>& colors,
  vector<Matrix4d>& cameras)
{
  Viewer3D::setup(points, colors, cameras);
}

Viewer3D::~Viewer3D() {
  finish_render_ = true;
  render_loop_.join();
  LOG(INFO) << "Render loop thread stop invoked";
}

void Viewer3D::setup(
  vector<Vector3d>& points,
  vector<Vector3d>& colors,
  vector<Matrix4d>& cameras)
{
  window_name = "SLAM: 3D";
  H = 900;
  W = 1200;
  finish_render_ = false;

  pangolin::CreateWindowAndBind(window_name, W, H);
  glEnable(GL_DEPTH_TEST);
  pangolin::GetBoundWindow()->RemoveCurrent();

  LOG(INFO) << "Starting render thread";
  render_loop_ = std::thread(&Viewer3D::update, this, std::ref(points), std::ref(colors), std::ref(cameras));
}

void Viewer3D::draw_camera(Matrix4d &Rt, cv::Point3f &color) {
  float w = 1.0;
  float h = w * 0.75;
  float z = w * 0.6;

  glPushMatrix();
  glMultMatrixd(Rt.data());

  glLineWidth(1.0f);
  glColor3f(color.x, color.y, color.z);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(w,h,z);
  glVertex3f(0,0,0);
  glVertex3f(w,-h,z);
  glVertex3f(0,0,0);
  glVertex3f(-w,-h,z);
  glVertex3f(0,0,0);
  glVertex3f(-w,h,z);

  glVertex3f(w,h,z);
  glVertex3f(w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(-w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(w,h,z);

  glVertex3f(-w,-h,z);
  glVertex3f(w,-h,z);
  glEnd();

  glPopMatrix();
}

void Viewer3D::update(
  vector<Vector3d>& points,  // in world
  vector<Vector3d>& colors,  // point color
  vector<Matrix4d>& cameras) // T_w_f
{
  pangolin::BindToContext(window_name);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  s_cam = pangolin::OpenGlRenderState(
    pangolin::ProjectionMatrix(W, H, 420, 420, W/2, H/2, 0.1, 1000),
    pangolin::ModelViewLookAt(0, 0, -10, // camera location in world
                              0,  0,  0, // where should the camera look at?
                              0, -1,  0) // camera y-axis
    );
  // s_cam = pangolin::OpenGlRenderState(
  //   pangolin::ProjectionMatrix(W, H, 420, 420, 512, 389, 0.1, 1000),
  //   pangolin::ModelViewLookAt(0, -10, 1, // camera location in world
  //                             0,  0,  0, // where should the camera look at?
  //                             0, -1,  0) // camera y-axis
  //   );

  pangolin::Handler3D handler(s_cam);
  d_cam = pangolin::CreateDisplay()
     .SetBounds(0.0, 1.0, 0.0, 1.0, -W/H)
     .SetHandler(&handler);

  cv::Point3f red, green, blue;
  red.x   = 1.0f; red.y   = 0.0f; red.z   = 0.0f;
  green.x = 0.0f; green.y = 1.0f; green.z = 0.0f;
  blue.x  = 0.0f; blue.y  = 0.0f; blue.z  = 1.0f;
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  const size_t num_points = points.size();
  float points_gl[3*num_points];
  for (size_t i=0; i<points.size(); ++i) {
    Vector3f point = points[i].cast<float>();
    for(size_t j=0;j<3;++j)
      *(points_gl+i*3+j) = point[j];
  }
  pangolin::GlBuffer glxyz(pangolin::GlArrayBuffer, num_points, GL_FLOAT, 3, GL_STATIC_DRAW);
  glxyz.Upload(points_gl, 3*sizeof(float)*num_points);

  while(!pangolin::ShouldQuit() && !finish_render_) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);

    // Draw world axes
    draw_axes();

    // Render cameras
    for (size_t i=0; i<cameras.size(); ++i) {
      draw_camera(cameras[i], green);
    }

    // Render points
    glPointSize(1.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    pangolin::RenderVbo(glxyz);

    // glBegin(GL_POINTS);
    // for (size_t i=0; i<points.size(); ++i) {
    //   Vector3f point = points[i].cast<float>();
    //   Vector3f color = colors[i].cast<float>();
    //   color /= 255.0;
    //   glColor3f(color(2), color(1), color(0)); // color is in BGR
    //   glVertex3f(point(0), point(1), point(2));
    // }
    glEnd();
    pangolin::FinishFrame();
  }
}

void Viewer3D::draw_axes() {
  glBegin(GL_LINES); 

  // x-axis
  glColor3f(1.0f, 0.0f, 0.0f);
  glVertex3f(0, 0, 0);
  glVertex3f(1, 0, 0);

  // y-axis
  glColor3f(0.0f, 1.0f, 0.0f);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 1, 0);

  // z-axis
  glColor3f(0.0f, 0.0f, 1.0f);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, 1);

  glEnd();
}

void Viewer3D::check_axes() {
  pangolin::BindToContext(window_name);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  s_cam = pangolin::OpenGlRenderState(
     pangolin::ProjectionMatrix(W, H, 420, 420, 512, 389, 0.1, 1000),
     pangolin::ModelViewLookAt(0, -1, -2, // camera location in world
                               0,  0,  0, // where should the camera look at?
                               0, -1,  0) // camera y-axis
     );

  pangolin::Handler3D handler(s_cam);
  d_cam = pangolin::CreateDisplay()
     .SetBounds(0.0, 1.0, 0.0, 1.0, -W/H)
     .SetHandler(&handler);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  while(!pangolin::ShouldQuit() && !finish_render_) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glBegin(GL_LINES);
    
    draw_axes();

    pangolin::FinishFrame();
  }
}

} // namespace df
