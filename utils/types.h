#pragma once

#include "config.h"
#include <opencv2/opencv.hpp>
#include <vector>

// Detection result with bounding box
struct alignas(float) Detection {
  float bbox[4];  // xmin ymin xmax ymax
  float conf;     // confidence score
  float class_id; // class id
};

// Segmentation result with mask
struct SegmentationResult {
  Detection det;            // Detection info (bbox, conf, class_id)
  cv::Mat mask;             // Segmentation mask for this object
  std::vector<float> mask_coeffs;  // Mask coefficients (32 values)
};