#include "ZedInterface.h"
#include <functional>

// ZED includes
#include <sl/Camera.hpp>

using namespace sl;

#ifdef WITH_ZED
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>

cv::Mat slMat2cvMat(sl::Mat &input);

ZedInterface::ZedInterface(int inWidth, int inHeight)
    : width(inWidth), height(inHeight) {
  // std::cout << "Resolution: " << inWidth << inHeight << std::cout;

  initSuccessful = false;

  latestDepthIndex.assign(-1);

  init_params.camera_resolution = RESOLUTION::VGA;
  // init_params.camera_resolution = RESOLUTION::HD720;
  // init_params.depth_mode = DEPTH_MODE::ULTRA;
  init_params.depth_mode = DEPTH_MODE::PERFORMANCE;
  init_params.coordinate_units = UNIT::METER; 

  // Open the camera
  sl::ERROR_CODE err = zed.open(init_params);
  if (err != sl::ERROR_CODE::SUCCESS) {
    printf("ZedInterface: %s\n", toString(err).c_str());
    closeZED();
  }

  // setAutoExposure(true);
  // setAutoWhiteBalance(true);

  for (int i = 0; i < numBuffers; i++) {
    uint8_t *newDepth = (uint8_t *)calloc(width * height * 2, sizeof(uint8_t));
    uint8_t *newImage = (uint8_t *)calloc(width * height * 3, sizeof(uint8_t));
    frameBuffers[i] = std::pair<std::pair<uint8_t *, uint8_t *>, int64_t>(
        std::pair<uint8_t *, uint8_t *>(newDepth, newImage), 0);
  }

  initSuccessful = true;

  startZED();
}

ZedInterface::~ZedInterface() {
  if (initSuccessful) {
    closeZED();

    for (int i = 0; i < numBuffers; i++) {
      free(frameBuffers[i].first.first);
      free(frameBuffers[i].first.second);
    }
  }
}

void ZedInterface::setAutoExposure(bool value) {
  zed.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, value);
}

void ZedInterface::setAutoWhiteBalance(bool value) {
  zed.setCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO, value);
}

bool ZedInterface::getAutoExposure() {
  return zed.getCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE) > 0;
}

bool ZedInterface::getAutoWhiteBalance() {
  return zed.getCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO) > 0;
}

void ZedInterface::startZED() {
  quit = false;
  while (zed.grab() != sl::ERROR_CODE::SUCCESS)
    sl::sleep_ms(1);
  zed_callback = std::thread(&ZedInterface::run, this);
}

void ZedInterface::closeZED() {
  quit = true;
  zed_callback.join();
  zed.close();
}

void ZedInterface::run() {

  // Set runtime parameters after opening the camera
  sl::RuntimeParameters runtime_parameters;
  runtime_parameters.sensing_mode =
      SENSING_MODE::STANDARD; // SENSING_MODE_FILL with stabilization

  int new_width = (int)width;
  int new_height = (int)height;

  // Prepare new image size to retrieve half-resolution images
  // Resolution image_size = zed.getCameraInformation().camera_resolution;
  // int new_width = image_size.width / 2;
  // int new_height = image_size.height / 2;

  Resolution new_image_size(new_width, new_height);

  sl::Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);

  sl::Mat depth_image_zed(new_width, new_height, MAT_TYPE::F32_C1);

  while (!quit) {
    //  /usr/local/zed/tools/'ZED Depth Viewer'

    if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {

      zed.retrieveImage(image_zed, sl::VIEW::LEFT, sl::MEM::CPU, new_image_size);
      // zed.retrieveImage(depth_image_zed, sl::VIEW_DEPTH, sl::MEM_CPU,
      // new_width, new_height);
      zed.retrieveMeasure(depth_image_zed, sl::MEASURE::DEPTH, sl::MEM::CPU,
                          new_image_size);

      lastDepthTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

      int bufferIndex = (latestDepthIndex.getValue() + 1) % numBuffers;

      frameBuffers[bufferIndex].second = lastDepthTime;

      //=========================================================================

      cv::Mat image_ocv = slMat2cvMat(image_zed);
      // cv::cvtColor(image_ocv, image_ocv, CV_RGBA2RGB);
      cv::cvtColor(image_ocv, image_ocv, CV_BGR2RGB);
      memcpy(frameBuffers[bufferIndex].first.second, image_ocv.data,
             new_width * new_height * 3);

      //=============================================================================

      cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
      cv::Size cvSize(new_width, new_height);
      cv::Mat depth(cvSize, CV_16UC1);
      depth_image_ocv *= 1000.0f;
      depth_image_ocv.convertTo(depth, CV_16UC1); // in mm, rounded
      memcpy(frameBuffers[bufferIndex].first.first, depth.data,
             new_width * new_height * 2);

      //=======================================================================

      cv::imshow("Image", image_ocv);
      cv::imshow("Depth", depth_image_ocv);
      cv::waitKey(3);

      latestDepthIndex++;

    } else
      sl::sleep_ms(1);
  }
}

cv::Mat slMat2cvMat(sl::Mat &mat) {
  if (mat.getMemoryType() == sl::MEM::GPU)
    mat.updateCPUfromGPU();

  int cv_type = -1;
  switch (mat.getDataType()) {
  case MAT_TYPE::F32_C1:
    cv_type = CV_32FC1;
    break;
  case MAT_TYPE::F32_C2:
    cv_type = CV_32FC2;
    break;
  case MAT_TYPE::F32_C3:
    cv_type = CV_32FC3;
    break;
  case MAT_TYPE::F32_C4:
    cv_type = CV_32FC4;
    break;
  case MAT_TYPE::U8_C1:
    cv_type = CV_8UC1;
    break;
  case MAT_TYPE::U8_C2:
    cv_type = CV_8UC2;
    break;
  case MAT_TYPE::U8_C3:
    cv_type = CV_8UC3;
    break;
  case MAT_TYPE::U8_C4:
    cv_type = CV_8UC4;
    break;
  default:
    break;
  }
  return cv::Mat((int)mat.getHeight(), (int)mat.getWidth(), cv_type,
                 mat.getPtr<sl::uchar1>(sl::MEM::CPU),
                 mat.getStepBytes(sl::MEM::CPU));
  // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer
  // from sl::Mat (getPtr<T>()) cv::Mat and sl::Mat will share a single memory
  // structure
  // return cv::Mat((int)mat.getHeight(), (int)mat.getWidth(), cv_type,
  //                mat.getPtr<sl::uchar1>(MEM::CPU));
}
#else

ZedInterface::ZedInterface(int inWidth, int inHeight)
    : width(inWidth), height(inHeight), initSuccessful(false) {
  errorText = "Compiled without stereolabs zed library";
}

ZedInterface::~ZedInterface() {}

void ZedInterface::setAutoExposure(bool value) {}

void ZedInterface::setAutoWhiteBalance(bool value) {}

bool ZedInterface::getAutoExposure() { return false; }

bool ZedInterface::getAutoWhiteBalance() { return false; }
#endif
