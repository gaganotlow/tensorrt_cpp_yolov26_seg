#pragma once

/* --------------------------------------------------------
 * YOLOv26 End2End Segmentation Configuration
 * - 端到端模型，输出已过NMS，无需后处理
 * - Bbox格式: [x1,y1,x2,y2,score,class_id,mask_coef*32]
 * --------------------------------------------------------*/

// For INT8, you need prepare the calibration dataset, please refer to
// https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5#int8-quantization
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "images";
const static char* kOutput0TensorName = "output0";  // Detection output or Mask prototype
const static char* kOutput1TensorName = "output1";  // Mask prototype or Detection output

// YOLOv26 End2End Segmentation model configuration
// Detection model and Segmentation model' number of classes
constexpr static int kNumClass = 2;  // 二分类: bg, tube
constexpr static int kNumMaskCoeff = 32;  // Number of mask coefficients
constexpr static int kMaskProtoH = 160;   // Mask prototype height
constexpr static int kMaskProtoW = 160;   // Mask prototype width

// Class names (2 classes total)
const static char* kClassNames[] = {
    "bg",     // 0 - background
    "tube"    // 1 - tube
};

// Class colors (BGR format for OpenCV)
const static int kClassColors[][3] = {
    {0, 0, 0},      // bg - black (usually not drawn)
    {0, 255, 0}     // tube - green
};

constexpr static int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// Maximum number of output detections (端到端模型输出的最大检测数)
constexpr static int kMaxNumOutputBbox = 300;

// YOLOv26 End2End output: 每个检测框包含 [x1,y1,x2,y2,score,class_id,mask_coef*32] = 38个值
constexpr static int kDetOutputDim = 6 + kNumMaskCoeff;  // 6 + 32 = 38

/* --------------------------------------------------------
 * These configs are NOT related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// 置信度阈值 (用于二次过滤，端到端模型内部已有NMS)
const static float kConfThresh = 0.3f;

// NMS阈值 (端到端模式下不起作用，但保留参数)
const static float kNmsThresh = 0.45f;

// bbox扩展比例 (用于mask裁剪时扩展bbox，避免截断mask边缘)
const static float kBboxExpandRatio = 0.1f;

// 是否启用mask裁剪
const static bool kEnableCropMask = true;

const static int kGpuId = 0;

// If your image size is larger than 4096 * 3112, please increase this value
const static int kMaxInputImageSize = 4096 * 3112;
