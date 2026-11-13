// Render textured Replica RGB and camera-z depth at the supplied fixed poses.
// Mirrors are excluded so RGB and geometry use the same physical surfaces.
#include <EGL.h>
#include <PTexLib.h>
#include "GLCheck.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>

int main(int argc, char** argv) {
  ASSERT(argc == 5 || argc == 7, "mesh.ply atlasFolder poses.tsv outputDirectory [exposureScale saturationScale]");
  const int w=512, h=512;
  EGLCtx egl; egl.PrintInformation();
  if (!checkGLVersion()) return 2;
  // RDF_TopLeft reverses screen Y relative to the SDK's BottomLeft example.
  // Reverse front-face winding too; keep the same physical surface faces.
  glFrontFace(GL_CW);
  PTexMesh mesh(argv[1], argv[2]);
  if (argc == 7) {
    const float exposureScale=std::strtof(argv[5],nullptr), saturationScale=std::strtof(argv[6],nullptr);
    ASSERT(std::isfinite(exposureScale) && exposureScale>0 && std::isfinite(saturationScale) && saturationScale>0,
           "Colour scales must be finite and positive");
    mesh.SetExposure(mesh.Exposure()*exposureScale);
    mesh.SetSaturation(mesh.Saturation()*saturationScale);
  }
  std::ofstream settings(std::string(argv[4])+"/sdk_colour_settings.txt");
  settings << mesh.Exposure() << " " << mesh.Gamma() << " " << mesh.Saturation() << "\n";
  pangolin::GlTexture colour(w,h), depth(w,h,GL_R32F,false,0,GL_RED,GL_FLOAT,0);
  pangolin::GlRenderBuffer renderBuffer(w,h);
  pangolin::GlFramebuffer cf(colour,renderBuffer), df(depth,renderBuffer);
  std::vector<unsigned char> rgb(w*h*3); std::vector<float> z(w*h);
  std::ifstream poses(argv[3]); std::string line;
  while (std::getline(poses,line)) {
    std::istringstream record(line); std::string id; record >> id;
    Eigen::Matrix4d transform;
    for (int r=0;r<4;r++) for(int c=0;c<4;c++) record >> transform(r,c);
    ASSERT(record && id.find('/')==std::string::npos, "Bad fixed pose");
    pangolin::OpenGlRenderState cam(
      // Pangolin principal points use pixel-corner coordinates. 256 corresponds
      // to pixel-index centre 255.5 for a 512 image. TopLeft means image-Y-down;
      // the raw GL download remains bottom-up and is flipped exactly once below.
      pangolin::ProjectionMatrixRDF_TopLeft(w,h,256.,256.,256.,256.,.1,100.), transform);
    cf.Bind(); glViewport(0,0,w,h); glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
    glEnable(GL_CULL_FACE); mesh.Render(cam); glDisable(GL_CULL_FACE); cf.Unbind();
    colour.Download(rgb.data(),GL_RGB,GL_UNSIGNED_BYTE);
    df.Bind(); glViewport(0,0,w,h); glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
    glEnable(GL_CULL_FACE); mesh.RenderDepth(cam,1.0f); glDisable(GL_CULL_FACE); df.Unbind();
    depth.Download(z.data(),GL_RED,GL_FLOAT);
    std::ofstream image(std::string(argv[4])+"/"+id+".ppm", std::ios::binary);
    image << "P6\n512 512\n255\n";
    std::ofstream raw(std::string(argv[4])+"/"+id+".f32", std::ios::binary);
    // OpenGL download is bottom-up. Export both RGB and depth in image-Y-down.
    for(int y=h-1;y>=0;y--) {
      image.write(reinterpret_cast<char*>(rgb.data()+y*w*3),w*3);
      raw.write(reinterpret_cast<char*>(z.data()+y*w),w*sizeof(float));
    }
  }
  return 0;
}
