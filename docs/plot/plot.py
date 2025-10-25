#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片合并脚本 - 将四张图片合成一张2x2的大图
按照文件首字母i，b，s，r顺序排序图片
"""

from PIL import Image
import os
import glob

def merge_images_2x2(input_dir=".", output_filename="comparison.png"):
    """
    将指定目录中的四张图片合并成一张2x2的大图，文件顺序为 i, b, s, r 首字母顺序
    
    Args:
        input_dir (str): 输入图片所在目录
        output_filename (str): 输出文件名
    """
    # 获取当前目录下的所有PNG图片
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    # 过滤掉输出文件本身
    image_files = [f for f in image_files if not f.endswith(output_filename)]

    # 自定义排序规则: 按 i, b, s, r 顺序
    def image_sort_key(filename):
        basename = os.path.basename(filename).lower()
        if basename.startswith('i'):
            return 0
        elif basename.startswith('b'):
            return 1
        elif basename.startswith('s'):
            return 2
        elif basename.startswith('r'):
            return 3
        else:
            return 99  # 保证其它图片排序在后面

    image_files = sorted(image_files, key=image_sort_key)

    # 按照 i, b, s, r 顺序查找
    expected_letters = ['i', 'b', 's', 'r']
    sorted_images = []
    for letter in expected_letters:
        found = False
        for f in image_files:
            if os.path.basename(f).lower().startswith(letter):
                sorted_images.append(f)
                found = True
                break
        if not found:
            print(f"错误：没有找到以'{letter}'开头的图片文件")
            print(f"当前所有图片：{[os.path.basename(f) for f in image_files]}")
            return
    image_files = sorted_images

    print(f"按照 i, b, s, r 顺序处理图片：")
    for i, img_file in enumerate(image_files):
        print(f"  {i+1}. {os.path.basename(img_file)}")
    
    # 打开所有图片
    images = []
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            images.append(img)
            print(f"成功加载：{os.path.basename(img_file)} - 尺寸：{img.size}")
        except Exception as e:
            print(f"加载图片失败 {os.path.basename(img_file)}: {e}")
            return
    
    # 获取所有图片的尺寸
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    # 计算合并后图片的尺寸
    max_width = max(widths)
    max_height = max(heights)
    
    # 创建新的空白图片 (2x2布局)
    merged_width = max_width * 2
    merged_height = max_height * 2
    merged_image = Image.new('RGB', (merged_width, merged_height), (255, 255, 255))
    
    print(f"合并图片尺寸：{merged_width} x {merged_height}")
    
    # 将图片放置到对应位置
    positions = [
        (0, 0),                    # 左上角 i
        (max_width, 0),            # 右上角 b
        (0, max_height),           # 左下角 s
        (max_width, max_height)    # 右下角 r
    ]
    
    for i, (img, pos) in enumerate(zip(images, positions)):
        # 居中放置
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        actual_pos = (pos[0] + x_offset, pos[1] + y_offset)
        merged_image.paste(img, actual_pos)
        print(f"已放置图片 {i+1} 到位置 {actual_pos}")
    
    # 保存合并后的图片
    output_path = os.path.join(input_dir, output_filename)
    merged_image.save(output_path)
    print(f"合并完成！输出文件：{output_path}")
    print(f"最终图片尺寸：{merged_image.size}")

if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== 图片合并工具 ===")
    print(f"工作目录：{script_dir}")
    
    # 执行合并
    merge_images_2x2(script_dir, "comparison.png")