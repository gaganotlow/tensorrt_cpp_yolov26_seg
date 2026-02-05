#!/usr/bin/env python3
"""
重新生成TensorRT引擎文件的脚本
用于解决版本不兼容问题
"""

import argparse
import os
import sys
from pathlib import Path

def check_tensorrt():
    """检查TensorRT是否可用"""
    try:
        import tensorrt as trt
        print(f"✓ TensorRT版本: {trt.__version__}")
        return True
    except ImportError:
        print("✗ TensorRT未安装")
        return False

def check_onnx_file(onnx_path):
    """检查ONNX文件是否存在"""
    if os.path.exists(onnx_path):
        print(f"✓ 找到ONNX文件: {onnx_path}")
        return True
    else:
        print(f"✗ ONNX文件不存在: {onnx_path}")
        return False

def convert_onnx_to_tensorrt(onnx_path, engine_path, precision='fp16', max_batch_size=1):
    """将ONNX模型转换为TensorRT引擎"""
    try:
        import tensorrt as trt
        
        # 创建TensorRT构建器
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        
        # 设置内存池大小 (TensorRT 8.0+的新API)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # 设置精度
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ 启用FP16精度")
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✓ 启用INT8精度")
        else:
            print("✓ 使用FP32精度")
        
        # 解析ONNX模型
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        print(f"解析ONNX文件: {onnx_path}")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("✗ 解析ONNX文件失败:")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        
        print("✓ ONNX文件解析成功")
        
        # 构建引擎
        print("开始构建TensorRT引擎...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("✗ 构建TensorRT引擎失败")
            return False
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✓ TensorRT引擎已保存到: {engine_path}")
        return True
        
    except Exception as e:
        print(f"✗ 转换过程中出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='重新生成TensorRT引擎文件')
    parser.add_argument('--onnx', help='ONNX模型文件路径')
    parser.add_argument('--engine', help='输出的TensorRT引擎文件路径')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                        help='推理精度 (默认: fp16)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='最大批处理大小 (默认: 1)')
    
    args = parser.parse_args()
    
    # 检查TensorRT
    if not check_tensorrt():
        sys.exit(1)
    
    # 如果没有提供参数，尝试自动查找
    if not args.onnx:
        # 查找可能的ONNX文件
        possible_onnx = [
            'igev_sceneflow_736x1280.onnx',
            'igev_model.onnx',
            'model.onnx'
        ]
        
        for onnx_file in possible_onnx:
            if os.path.exists(onnx_file):
                args.onnx = onnx_file
                break
        
        if not args.onnx:
            print("未找到ONNX文件，请使用 --onnx 参数指定")
            print("可能的解决方案:")
            print("1. 首先需要将PyTorch模型导出为ONNX格式")
            print("2. 然后使用此脚本将ONNX转换为TensorRT引擎")
            sys.exit(1)
    
    if not args.engine:
        # 根据ONNX文件名生成引擎文件名
        onnx_path = Path(args.onnx)
        args.engine = str(onnx_path.with_suffix('.engine'))
    
    # 检查ONNX文件
    if not check_onnx_file(args.onnx):
        sys.exit(1)
    
    # 转换
    print(f"\n开始转换:")
    print(f"  输入ONNX: {args.onnx}")
    print(f"  输出引擎: {args.engine}")
    print(f"  精度: {args.precision}")
    print(f"  批处理大小: {args.batch_size}")
    print()
    
    success = convert_onnx_to_tensorrt(
        args.onnx, 
        args.engine, 
        args.precision, 
        args.batch_size
    )
    
    if success:
        print(f"\n✓ 转换完成! 新的TensorRT引擎文件: {args.engine}")
        print("现在可以使用新的引擎文件进行推理了")
    else:
        print(f"\n✗ 转换失败")
        sys.exit(1)

if __name__ == '__main__':
    main()