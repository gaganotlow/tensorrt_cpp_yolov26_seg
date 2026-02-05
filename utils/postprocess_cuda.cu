#include "postprocess_cuda.h"
#include "cuda_utils.h"
#include "config.h"
#include <cmath>
#include <algorithm>

// ============= CUDA Kernels =============

// Mask 原型矩阵乘法并应用 sigmoid
__global__ void mask_decode_kernel(
    const float* mask_proto,  // [32, 160, 160]
    const float* mask_coeff,  // [N, 32]
    float* output_masks,      // [N, 160, 160]
    int num_masks,
    int proto_h,
    int proto_w)
{
    const int n = blockIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (n >= num_masks || y >= proto_h || x >= proto_w) return;
    
    float sum = 0.0f;
    for (int k = 0; k < 32; ++k) {
        sum += mask_coeff[n*32 + k] * mask_proto[k*proto_h*proto_w + y*proto_w + x];
    }
    // Apply sigmoid
    output_masks[n*proto_h*proto_w + y*proto_w + x] = 1.0f / (1.0f + expf(-sum));
}

// Crop mask by bounding box with optional expansion
__global__ void crop_mask_kernel(
    float* masks,        // [N, proto_h, proto_w]
    float* boxes,        // [N, 4]: x1, y1, x2, y2 in proto coordinate
    int num_masks,
    int proto_h,
    int proto_w,
    float expand_ratio)
{
    int n = blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (n >= num_masks || y >= proto_h || x >= proto_w) return;
    
    float x1 = boxes[n * 4 + 0];
    float y1 = boxes[n * 4 + 1];
    float x2 = boxes[n * 4 + 2];
    float y2 = boxes[n * 4 + 3];
    
    // Apply expansion
    if (expand_ratio > 0.0f) {
        float box_w = x2 - x1;
        float box_h = y2 - y1;
        float expand_w = box_w * expand_ratio / 2.0f;
        float expand_h = box_h * expand_ratio / 2.0f;
        
        x1 = fmaxf(0.0f, x1 - expand_w);
        y1 = fmaxf(0.0f, y1 - expand_h);
        x2 = fminf((float)proto_w, x2 + expand_w);
        y2 = fminf((float)proto_h, y2 + expand_h);
    }
    
    int idx = n * proto_h * proto_w + y * proto_w + x;
    
    // Check if pixel is outside bounding box
    if (x < x1 || x >= x2 || y < y1 || y >= y2) {
        masks[idx] = 0.0f;
    }
}

// ============= Host Functions =============

void cuda_mask_decode(
    float* mask_proto_device,
    float* mask_coeff_device,
    float* output_masks_device,
    int num_masks,
    int proto_h,
    int proto_w)
{
    dim3 block(1, 16, 16);
    dim3 grid(
        num_masks,
        (proto_h + block.y - 1) / block.y,
        (proto_w + block.z - 1) / block.z
    );
    
    mask_decode_kernel<<<grid, block>>>(
        mask_proto_device, mask_coeff_device, output_masks_device,
        num_masks, proto_h, proto_w);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_crop_mask(
    float* masks_device,
    float* boxes_device,
    int num_masks,
    int proto_h,
    int proto_w,
    float expand_ratio)
{
    dim3 block(1, 16, 16);
    dim3 grid(
        num_masks,
        (proto_h + block.y - 1) / block.y,
        (proto_w + block.z - 1) / block.z
    );
    
    crop_mask_kernel<<<grid, block>>>(
        masks_device, boxes_device,
        num_masks, proto_h, proto_w, expand_ratio);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============= High-level Processing Functions for YOLOv26 End2End =============

void decode_yolov26_e2e_detection(
    std::vector<Detection>& detections,
    std::vector<std::vector<float>>& mask_coeffs_out,
    float* det_output,
    int num_dets,
    float conf_thresh)
{
    detections.clear();
    mask_coeffs_out.clear();
    
    // YOLOv26 端到端输出格式: [N, 38]
    // 每行: [x1, y1, x2, y2, score, class_id, mask_coef*32]
    for (int i = 0; i < num_dets; i++) {
        float* row = det_output + i * kDetOutputDim;
        
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];
        float score = row[4];
        float class_id = row[5];
        
        // Filter by confidence threshold
        if (score < conf_thresh) continue;
        
        // 跳过背景类 (class_id == 0 是 bg)
        // 如果需要保留背景类，可以注释掉这行
        // if (class_id == 0) continue;
        
        // Create detection
        Detection det;
        det.bbox[0] = x1;
        det.bbox[1] = y1;
        det.bbox[2] = x2;
        det.bbox[3] = y2;
        det.conf = score;
        det.class_id = class_id;
        
        detections.push_back(det);
        
        // Extract mask coefficients (32 values)
        std::vector<float> coeffs(kNumMaskCoeff);
        for (int k = 0; k < kNumMaskCoeff; k++) {
            coeffs[k] = row[6 + k];
        }
        mask_coeffs_out.push_back(coeffs);
    }
}

void generate_segmentation_masks_gpu(
    std::vector<SegmentationResult>& seg_results,
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& mask_coeffs,
    float* mask_protos,
    int proto_h,
    int proto_w,
    int img_h,
    int img_w)
{
    seg_results.clear();
    
    if (detections.empty()) return;
    
    int num_masks = detections.size();
    
    // Calculate letterbox parameters
    float scale = std::min(kInputW / float(img_w), kInputH / float(img_h));
    int new_w = int(img_w * scale);
    int new_h = int(img_h * scale);
    int offset_x = (kInputW - new_w) / 2;
    int offset_y = (kInputH - new_h) / 2;
    
    // Prepare mask coefficients: [N, 32]
    std::vector<float> coeffs_flat(num_masks * 32);
    for (int i = 0; i < num_masks; i++) {
        for (int k = 0; k < 32; k++) {
            coeffs_flat[i * 32 + k] = mask_coeffs[i][k];
        }
    }
    
    // Allocate GPU memory
    float* proto_device;
    float* coeffs_device;
    float* masks_device;
    
    CUDA_CHECK(cudaMalloc(&proto_device, 32 * proto_h * proto_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coeffs_device, num_masks * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&masks_device, num_masks * proto_h * proto_w * sizeof(float)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(proto_device, mask_protos, 
                          32 * proto_h * proto_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(coeffs_device, coeffs_flat.data(), 
                          num_masks * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Generate masks on GPU
    cuda_mask_decode(proto_device, coeffs_device, masks_device, 
                     num_masks, proto_h, proto_w);
    
    // Crop masks by bounding boxes (in proto coordinates) if enabled
    if (kEnableCropMask) {
        std::vector<float> boxes_proto(num_masks * 4);
        float width_ratio = proto_w / float(kInputW);
        float height_ratio = proto_h / float(kInputH);
        
        for (int i = 0; i < num_masks; i++) {
            boxes_proto[i * 4 + 0] = detections[i].bbox[0] * width_ratio;   // x1
            boxes_proto[i * 4 + 1] = detections[i].bbox[1] * height_ratio;  // y1
            boxes_proto[i * 4 + 2] = detections[i].bbox[2] * width_ratio;   // x2
            boxes_proto[i * 4 + 3] = detections[i].bbox[3] * height_ratio;  // y2
        }
        
        float* boxes_device;
        CUDA_CHECK(cudaMalloc(&boxes_device, num_masks * 4 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(boxes_device, boxes_proto.data(), 
                              num_masks * 4 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Crop masks with expansion
        cuda_crop_mask(masks_device, boxes_device, num_masks, proto_h, proto_w, kBboxExpandRatio);
        
        CUDA_CHECK(cudaFree(boxes_device));
    }
    
    // Copy masks back to host
    std::vector<float> masks_host(num_masks * proto_h * proto_w);
    CUDA_CHECK(cudaMemcpy(masks_host.data(), masks_device, 
                          num_masks * proto_h * proto_w * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    // Process each mask
    for (int i = 0; i < num_masks; i++) {
        SegmentationResult seg;
        seg.det = detections[i];
        seg.mask_coeffs = mask_coeffs[i];
        
        // Convert to cv::Mat
        cv::Mat mask_proto(proto_h, proto_w, CV_32F, 
                          &masks_host[i * proto_h * proto_w]);
        
        // Step 1: Resize mask from 160x160 to 640x640
        cv::Mat mask_input;
        cv::resize(mask_proto, mask_input, cv::Size(kInputW, kInputH), 
                   0, 0, cv::INTER_LINEAR);
        
        // Step 2: Crop the valid region (remove letterbox padding)
        int crop_x = std::max(0, std::min(offset_x, kInputW - 1));
        int crop_y = std::max(0, std::min(offset_y, kInputH - 1));
        int crop_w = std::max(1, std::min(new_w, kInputW - crop_x));
        int crop_h = std::max(1, std::min(new_h, kInputH - crop_y));
        
        cv::Rect crop_rect(crop_x, crop_y, crop_w, crop_h);
        cv::Mat mask_cropped = mask_input(crop_rect);
        
        // Step 3: Resize to original image size
        cv::resize(mask_cropped, seg.mask, cv::Size(img_w, img_h), 
                   0, 0, cv::INTER_LINEAR);
        
        // Step 4: Threshold mask
        cv::threshold(seg.mask, seg.mask, 0.5, 1.0, cv::THRESH_BINARY);
        
        seg_results.push_back(seg);
    }
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(proto_device));
    CUDA_CHECK(cudaFree(coeffs_device));
    CUDA_CHECK(cudaFree(masks_device));
}

void process_yolov26_e2e_segmentation(
    std::vector<SegmentationResult>& results,
    float* output0,
    float* output1,
    int output0_dim0,
    int output0_dim1,
    int output1_dim0,
    int output1_dim1,
    int img_h,
    int img_w,
    float conf_thresh)
{
    float* proto = nullptr;
    float* det = nullptr;
    int num_dets = 0;
    
    // 自动判断哪个是proto，哪个是det
    // proto形状: [1, 32, 160, 160] 或 [32, 160, 160] -> 第一维是32 (mask channel)
    // det形状: [N, 38] -> 第二维是38 (bbox + score + class + mask_coef)
    
    // 判断逻辑：检查维度来确定proto和det
    // output0: dim0=第一维, dim1=第二维
    // 如果dim1 == 38 (kDetOutputDim)，则为det
    // 如果dim0 == 32，则为proto
    
    bool output0_is_det = (output0_dim1 == kDetOutputDim);
    bool output1_is_det = (output1_dim1 == kDetOutputDim);
    
    if (output0_is_det && !output1_is_det) {
        // output0 是检测结果，output1 是proto
        det = output0;
        proto = output1;
        num_dets = output0_dim0;
    } else if (!output0_is_det && output1_is_det) {
        // output0 是proto，output1 是检测结果
        proto = output0;
        det = output1;
        num_dets = output1_dim0;
    } else {
        // 备用判断：如果dim0 == 32，则为proto
        if (output0_dim0 == 32) {
            proto = output0;
            det = output1;
            num_dets = output1_dim0;
        } else if (output1_dim0 == 32) {
            proto = output1;
            det = output0;
            num_dets = output0_dim0;
        } else {
            std::cerr << "无法确定output0和output1的类型!" << std::endl;
            std::cerr << "output0: [" << output0_dim0 << ", " << output0_dim1 << "]" << std::endl;
            std::cerr << "output1: [" << output1_dim0 << ", " << output1_dim1 << "]" << std::endl;
            results.clear();
            return;
        }
    }
    
    // Step 1: 解析端到端检测输出 (已过NMS)
    std::vector<Detection> detections;
    std::vector<std::vector<float>> mask_coeffs;
    decode_yolov26_e2e_detection(detections, mask_coeffs, det, num_dets, conf_thresh);
    
    if (detections.empty()) {
        results.clear();
        return;
    }
    
    // Step 2: 生成分割mask
    generate_segmentation_masks_gpu(results, detections, mask_coeffs, proto, 
                                     kMaskProtoH, kMaskProtoW, img_h, img_w);
}

// ============= 绘图函数 =============

static float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

static cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float scale = std::min(kInputH / float(img.cols), kInputW / float(img.rows));
    int offsetx = (kInputW - img.cols * scale) / 2; 
    int offsety = (kInputH - img.rows * scale) / 2; 

    size_t output_width = img.cols;
    size_t output_height = img.rows;

    float x1 = (bbox[0] - offsetx) / scale;
    float y1 = (bbox[1] - offsety) / scale;
    float x2 = (bbox[2] - offsetx) / scale;
    float y2 = (bbox[3] - offsety) / scale;

    x1 = clamp(x1, 0, output_width);
    y1 = clamp(y1, 0, output_height);
    x2 = clamp(x2, 0, output_width);
    y2 = clamp(y2, 0, output_height);

    auto left = x1;
    auto width = clamp(x2 - x1, 0, output_width);
    auto top = y1;
    auto height = clamp(y2 - y1, 0, output_height);

    return cv::Rect(left, top, width, height);
}

void draw_segmentation_results(cv::Mat& img, const std::vector<SegmentationResult>& results)
{
    for (const auto& result : results) {
        int class_id = static_cast<int>(result.det.class_id);
        
        // 使用配置中的颜色
        cv::Scalar color;
        if (class_id >= 0 && class_id < kNumClass) {
            color = cv::Scalar(kClassColors[class_id][0], 
                              kClassColors[class_id][1], 
                              kClassColors[class_id][2]);
        } else {
            color = cv::Scalar(255, 255, 255);  // 默认白色
        }
        
        // Draw segmentation mask
        cv::Mat mask_resized;
        result.mask.convertTo(mask_resized, CV_8UC1, 255);
        
        // Blend mask with image (半透明效果)
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (mask_resized.at<uchar>(y, x) > 127) {
                    img.at<cv::Vec3b>(y, x) = 0.5 * img.at<cv::Vec3b>(y, x) + 
                                               0.5 * cv::Vec3b(color[0], color[1], color[2]);
                }
            }
        }
        
        // Draw bounding box
        cv::Rect r = get_rect(img, (float*)result.det.bbox);
        cv::rectangle(img, r, color, 2, cv::LINE_AA);
        
        // Draw class label with class name
        std::string class_name = (class_id >= 0 && class_id < kNumClass) ? 
                                 kClassNames[class_id] : "unknown";
        std::string label = class_name + " " + std::to_string(int(result.det.conf * 100)) + "%";
        
        // 绘制标签背景
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point text_org(r.x, r.y - 5);
        if (text_org.y < text_size.height) {
            text_org.y = r.y + r.height + text_size.height + 5;
        }
        cv::rectangle(img, 
                     cv::Point(text_org.x, text_org.y - text_size.height - 2),
                     cv::Point(text_org.x + text_size.width, text_org.y + 2),
                     color, -1, cv::LINE_AA);
        cv::putText(img, label, text_org, 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}
