# YOLOv26 End2End TensorRT C++ 分割推理

基于 TensorRT 8 的 YOLOv26 端到端实例分割模型 C++ 推理实现。

## 特性

- **端到端模型**: YOLOv26 输出已过 NMS，无需后处理
- **GPU 加速**: 预处理和后处理均使用 CUDA 加速
- **简化流程**: 相比传统 YOLO，后处理逻辑大幅简化
- **实例分割**: 支持输出每个检测对象的分割 mask

## 模型输出格式

YOLOv26 端到端模型的检测输出格式为:
```
[x1, y1, x2, y2, score, class_id, mask_coef*32]
```
- `x1, y1, x2, y2`: 边界框坐标 (已转换为 xyxy 格式)
- `score`: 置信度分数
- `class_id`: 类别 ID
- `mask_coef*32`: 32 个 mask 系数

## 目录结构

```
.
├── CMakeLists.txt          # CMake 配置文件
├── build.cu                # 构建 TensorRT 引擎
├── runtime.cu              # 运行时推理
├── utils/
│   ├── config.h            # 模型配置参数
│   ├── types.h             # 数据结构定义
│   ├── preprocess.cu/h     # CUDA 预处理
│   ├── postprocess_cuda.cu/h  # CUDA 后处理
│   └── cuda_utils.h        # CUDA 工具函数
├── weights/                # 模型文件目录
├── media/                  # 测试图片目录
├── output/                 # 输出结果目录
└── cmake/                  # CMake 模块
```

## 环境依赖

- CUDA >= 11.0
- TensorRT 8.x
- OpenCV >= 4.0
- CMake >= 3.11

## 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 使用方法

### 1. 构建 TensorRT 引擎

从 ONNX 模型构建 FP16 引擎:
```bash
./build /path/to/your/yolov26_seg.onnx
```

构建 INT8 引擎 (需要校准数据):
```bash
./build /path/to/your/yolov26_seg.onnx /path/to/calib_images /path/to/calib_list.txt
```

引擎文件将保存到 `./weights/yolov26_e2e_seg.engine`

### 2. 运行推理

```bash
./runtime ./weights/yolov26_e2e_seg.engine /path/to/image.jpg [output_path]
```

示例:
```bash
./runtime ./weights/yolov26_e2e_seg.engine ./media/test.jpg ./output/result.jpg
```

## 配置参数

在 `utils/config.h` 中可以修改以下参数:

```cpp
// 类别配置
constexpr static int kNumClass = 2;  // 二分类: bg, tube
const static char* kClassNames[] = {"bg", "tube"};

// 输入尺寸
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// 置信度阈值 (用于二次过滤)
const static float kConfThresh = 0.3f;

// bbox扩展比例 (用于mask裁剪)
const static float kBboxExpandRatio = 0.1f;

// 是否启用mask裁剪
const static bool kEnableCropMask = true;
```

## 性能

程序会自动运行 10 次推理并输出性能统计:
- 预处理时间
- 推理时间
- 后处理时间
- 总时间和 FPS

## 与 YOLOv11 的主要区别

| 特性 | YOLOv11 | YOLOv26 E2E |
|------|---------|-------------|
| NMS | 需要后处理 | 模型内置 |
| 输出格式 | [4+num_classes+32, 8400] | [N, 38] |
| bbox 格式 | cx, cy, w, h | x1, y1, x2, y2 |
| 后处理复杂度 | 高 | 低 |