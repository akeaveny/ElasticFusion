#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <map>

#ifdef WITH_ZED
#include <sl/Camera.hpp>
#endif

#include "CameraInterface.h"

class ZedInterface : public CameraInterface
{
public:
  ZedInterface(int width = 640,int height = 480);
  virtual ~ZedInterface();

  const int width,height;   
  bool getAutoExposure();
  bool getAutoWhiteBalance();
  virtual void setAutoExposure(bool value);
  virtual void setAutoWhiteBalance(bool value);

  virtual bool ok()
  {
    return initSuccessful;
  }

  virtual std::string error()
  {
    return errorText;
  }

private:

#ifdef WITH_ZED
  void startZED();
  void closeZED();
  void run();   
  bool quit;
  sl::Camera zed;
  sl::InitParameters init_params;  
  int64_t lastDepthTime;  
  std::thread  zed_callback; 
#endif
  
  bool initSuccessful = true; 
  std::string errorText; 
 

  };
  

