#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <vector>

// ============= CUDA-accelerated YOLOv26 End2End Segmentation Postprocessing =============
// YOLOv26 是端到端模型，输出已过NMS，无需NMS后处理
// 输出格式: [x1,y1,x2,y2,score,class_id,mask_coef*32]

// GPU-accelerated mask decoding
// mask_proto_device: [32, proto_h, proto_w] on GPU
// mask_coeff_device: [N, 32] on GPU
// output_masks_device: [N, proto_h, proto_w] on GPU
void cuda_mask_decode(
    float* mask_proto_device,
    float* mask_coeff_device,
    float* output_masks_device,
    int num_masks,
    int proto_h,
    int proto_w);

// Crop masks by bounding boxes on GPU
void cuda_crop_mask(
    float* masks_device,
    float* boxes_device,
    int num_masks,
    int proto_h,
    int proto_w,
    float expand_ratio = 0.1f);

// ============= High-level Processing Functions for YOLOv26 End2End =============

// 解析YOLOv26端到端检测输出 (已过NMS)
// det_output: [N, 38] on HOST - 每行为 [x1,y1,x2,y2,score,class_id,mask_coef*32]
// num_dets: 检测数量
void decode_yolov26_e2e_detection(
    std::vector<Detection>& detections,
    std::vector<std::vector<float>>& mask_coeffs_out,
    float* det_output,
    int num_dets,
    float conf_thresh);

// Generate segmentation masks using GPU acceleration
void generate_segmentation_masks_gpu(
    std::vector<SegmentationResult>& seg_results,
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& mask_coeffs,
    float* mask_protos,  // [1, 32, 160, 160] or [32, 160, 160] on HOST
    int proto_h,         // 160
    int proto_w,         // 160
    int img_h,           // Original image height
    int img_w);          // Original image width

// Main processing function for YOLOv26 End2End segmentation
// 这是YOLOv26端到端模型的主处理函数
void process_yolov26_e2e_segmentation(
    std::vector<SegmentationResult>& results,
    float* output0,      // 输出0 (可能是proto或det)
    float* output1,      // 输出1 (可能是det或proto)
    int output0_dim0,    // output0第一维大小
    int output0_dim1,    // output0第二维大小
    int output1_dim0,    // output1第一维大小
    int output1_dim1,    // output1第二维大小
    int img_h,
    int img_w,
    float conf_thresh);

// ============= 绘图函数 =============

// Draw segmentation results on image
void draw_segmentation_results(cv::Mat& img, const std::vector<SegmentationResult>& results);
