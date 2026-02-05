#!/usr/bin/env python3
"""
修复ONNX模型中TensorRT 8不支持的Mod操作符
将 Mod 替换为等效的 Sub/Div/Mul 操作
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import argparse
import os

def replace_mod_with_equivalent(model):
    """
    将 Mod 操作替换为等效操作: a % b = a - (a // b) * b
    对于整数: a % b = a - floor(a / b) * b
    """
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    
    for i, node in enumerate(graph.node):
        if node.op_type == "Mod":
            print(f"找到 Mod 节点: {node.name}")
            print(f"  输入: {node.input}")
            print(f"  输出: {node.output}")
            
            input_a = node.input[0]
            input_b = node.input[1]
            output = node.output[0]
            
            # 创建中间节点名称
            base_name = node.name if node.name else f"mod_replace_{i}"
            div_output = f"{base_name}_div_output"
            floor_output = f"{base_name}_floor_output"
            mul_output = f"{base_name}_mul_output"
            
            # 1. Div: a / b
            div_node = helper.make_node(
                "Div",
                inputs=[input_a, input_b],
                outputs=[div_output],
                name=f"{base_name}_div"
            )
            
            # 2. Floor: floor(a / b)
            floor_node = helper.make_node(
                "Floor",
                inputs=[div_output],
                outputs=[floor_output],
                name=f"{base_name}_floor"
            )
            
            # 3. Mul: floor(a / b) * b
            mul_node = helper.make_node(
                "Mul",
                inputs=[floor_output, input_b],
                outputs=[mul_output],
                name=f"{base_name}_mul"
            )
            
            # 4. Sub: a - floor(a / b) * b
            sub_node = helper.make_node(
                "Sub",
                inputs=[input_a, mul_output],
                outputs=[output],
                name=f"{base_name}_sub"
            )
            
            nodes_to_remove.append(node)
            nodes_to_add.extend([div_node, floor_node, mul_node, sub_node])
            
            print(f"  已替换为: Div -> Floor -> Mul -> Sub")
    
    # 移除旧节点，添加新节点
    for node in nodes_to_remove:
        graph.node.remove(node)
    
    # 找到正确的插入位置
    for new_node in nodes_to_add:
        graph.node.append(new_node)
    
    return model, len(nodes_to_remove)

def replace_mod_int_with_equivalent(model):
    """
    针对整数类型的 Mod 操作，使用 Cast 确保类型正确
    a % b = a - Cast(floor(Cast(a, float) / Cast(b, float)), int) * b
    """
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    
    for i, node in enumerate(graph.node):
        if node.op_type == "Mod":
            print(f"找到 Mod 节点: {node.name}")
            print(f"  输入: {node.input}")
            print(f"  输出: {node.output}")
            
            input_a = node.input[0]
            input_b = node.input[1]
            output = node.output[0]
            
            # 创建中间节点名称
            base_name = node.name.replace("/", "_") if node.name else f"mod_replace_{i}"
            
            cast_a_output = f"{base_name}_cast_a"
            cast_b_output = f"{base_name}_cast_b"
            div_output = f"{base_name}_div"
            floor_output = f"{base_name}_floor"
            cast_back_output = f"{base_name}_cast_back"
            mul_output = f"{base_name}_mul"
            
            # 1. Cast a to float
            cast_a_node = helper.make_node(
                "Cast",
                inputs=[input_a],
                outputs=[cast_a_output],
                name=f"{base_name}_cast_a",
                to=TensorProto.FLOAT
            )
            
            # 2. Cast b to float  
            cast_b_node = helper.make_node(
                "Cast",
                inputs=[input_b],
                outputs=[cast_b_output],
                name=f"{base_name}_cast_b",
                to=TensorProto.FLOAT
            )
            
            # 3. Div: a / b (float)
            div_node = helper.make_node(
                "Div",
                inputs=[cast_a_output, cast_b_output],
                outputs=[div_output],
                name=f"{base_name}_div"
            )
            
            # 4. Floor: floor(a / b)
            floor_node = helper.make_node(
                "Floor",
                inputs=[div_output],
                outputs=[floor_output],
                name=f"{base_name}_floor"
            )
            
            # 5. Cast back to int32
            cast_back_node = helper.make_node(
                "Cast",
                inputs=[floor_output],
                outputs=[cast_back_output],
                name=f"{base_name}_cast_back",
                to=TensorProto.INT32
            )
            
            # 6. Mul: floor(a / b) * b
            mul_node = helper.make_node(
                "Mul",
                inputs=[cast_back_output, input_b],
                outputs=[mul_output],
                name=f"{base_name}_mul"
            )
            
            # 7. Sub: a - floor(a / b) * b
            sub_node = helper.make_node(
                "Sub",
                inputs=[input_a, mul_output],
                outputs=[output],
                name=f"{base_name}_sub"
            )
            
            nodes_to_remove.append(node)
            nodes_to_add.extend([cast_a_node, cast_b_node, div_node, floor_node, 
                                cast_back_node, mul_node, sub_node])
            
            print(f"  已替换为: Cast -> Div -> Floor -> Cast -> Mul -> Sub")
    
    # 移除旧节点
    for node in nodes_to_remove:
        graph.node.remove(node)
    
    # 添加新节点
    for new_node in nodes_to_add:
        graph.node.append(new_node)
    
    return model, len(nodes_to_remove)

def fix_onnx_model(input_path, output_path):
    """修复ONNX模型"""
    print(f"加载ONNX模型: {input_path}")
    model = onnx.load(input_path)
    
    print("\n检查并替换不支持的操作符...")
    model, num_replaced = replace_mod_int_with_equivalent(model)
    
    if num_replaced > 0:
        print(f"\n共替换了 {num_replaced} 个 Mod 操作")
        
        # 验证模型
        print("\n验证修改后的模型...")
        try:
            onnx.checker.check_model(model)
            print("✓ 模型验证通过")
        except Exception as e:
            print(f"⚠ 模型验证警告 (可能仍然可用): {e}")
        
        # 保存模型
        print(f"\n保存修复后的模型: {output_path}")
        onnx.save(model, output_path)
        print("✓ 完成!")
    else:
        print("未找到需要替换的 Mod 操作")
    
    return num_replaced > 0

def main():
    parser = argparse.ArgumentParser(description='修复ONNX模型中TensorRT不支持的操作符')
    parser.add_argument('--input', '-i', required=True, help='输入ONNX文件路径')
    parser.add_argument('--output', '-o', help='输出ONNX文件路径 (默认: 输入文件名_fixed.onnx)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_fixed{ext}"
    
    fix_onnx_model(args.input, args.output)

if __name__ == "__main__":
    main()
