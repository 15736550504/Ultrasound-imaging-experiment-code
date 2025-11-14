import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd


def test_inference_speed(model_path, device='cuda', warmup=10, num_tests=100, img_size=640):
    """
    测试模型推理速度

    参数:
    model_path: 模型路径
    device: 推理设备 ('cuda' 或 'cpu')
    warmup: 预热迭代次数
    num_tests: 正式测试迭代次数
    img_size: 输入图像尺寸

    返回:
    fps: 平均帧率 (FPS)
    latency: 平均延迟 (毫秒)
    """
    # 加载模型
    model = YOLO(model_path)

    # 设置设备
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # 创建测试图像 (随机生成)
    img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # 预热阶段
    print(f"预热中 ({warmup} 次迭代)...")
    for _ in range(warmup):
        _ = model(img, device=device, verbose=False)

    # 正式测试
    print(f"正式测试中 ({num_tests} 次迭代)...")
    start_time = time.time()

    for _ in range(num_tests):
        _ = model(img, device=device, verbose=False)

    # 计算性能指标
    total_time = time.time() - start_time
    latency = total_time / num_tests * 1000  # 毫秒
    fps = num_tests / total_time

    print(f"设备: {device.upper()}")
    print(f"测试次数: {num_tests}")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"平均延迟: {latency:.2f} 毫秒")
    print(f"推理帧率: {fps:.2f} FPS")

    return fps, latency


def visualize_results(gpu_results, cpu_results):
    """可视化测试结果"""
    plt.figure(figsize=(12, 6))

    # FPS 对比
    plt.subplot(1, 2, 1)
    devices = ['GPU', 'CPU']
    fps_values = [gpu_results['fps'], cpu_results['fps']]
    plt.bar(devices, fps_values, color=['blue', 'green'])
    plt.title('推理帧率 (FPS) 对比')
    plt.ylabel('帧率 (FPS)')
    for i, v in enumerate(fps_values):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')

    # 延迟对比
    plt.subplot(1, 2, 2)
    latency_values = [gpu_results['latency'], cpu_results['latency']]
    plt.bar(devices, latency_values, color=['blue', 'green'])
    plt.title('推理延迟对比')
    plt.ylabel('延迟 (毫秒)')
    for i, v in enumerate(latency_values):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig('inference_performance.png')
    plt.show()


def save_results_to_csv(gpu_results, cpu_results, filename='inference_results.csv'):
    """保存结果到CSV文件"""
    data = {
        '设备': ['GPU', 'CPU'],
        'FPS': [gpu_results['fps'], cpu_results['fps']],
        '延迟(ms)': [gpu_results['latency'], cpu_results['latency']],
        '测试次数': [gpu_results['num_tests'], cpu_results['num_tests']],
        '模型路径': [gpu_results['model_path'], cpu_results['model_path']]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"结果已保存到 {filename}")


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = r"/home/link/Yang/LHL/超声/腹部/my_project/腹部data6.23/yolov8n-UFEA best训练 b32/weights/epoch70.pt"  # 替换为你的模型路径
    WARMUP = 30  # 预热迭代次数
    NUM_TESTS = 100  # 正式测试迭代次数
    IMG_SIZE = 640  # 输入图像尺寸

    print("=" * 50)
    print("YOLOv8 模型推理性能测试")
    print("=" * 50)
    print(f"模型: {MODEL_PATH}")
    print(f"预热迭代: {WARMUP}")
    print(f"测试迭代: {NUM_TESTS}")
    print(f"图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print("\n")

    # 检查GPU是否可用
    if torch.cuda.is_available():
        print("检测到可用GPU")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到GPU，将使用CPU测试")

    print("\n")

    # GPU测试
    print("开始GPU测试...")
    gpu_fps, gpu_latency = test_inference_speed(
        MODEL_PATH,
        device='cuda',
        warmup=WARMUP,
        num_tests=NUM_TESTS,
        img_size=IMG_SIZE
    )

    gpu_results = {
        'device': 'GPU',
        'fps': gpu_fps,
        'latency': gpu_latency,
        'num_tests': NUM_TESTS,
        'model_path': MODEL_PATH
    }

    print("\nGPU测试完成!\n")

    # CPU测试
    print("开始CPU测试...")
    cpu_fps, cpu_latency = test_inference_speed(
        MODEL_PATH,
        device='cpu',
        warmup=WARMUP,
        num_tests=NUM_TESTS,
        img_size=IMG_SIZE
    )

    cpu_results = {
        'device': 'CPU',
        'fps': cpu_fps,
        'latency': cpu_latency,
        'num_tests': NUM_TESTS,
        'model_path': MODEL_PATH
    }

    print("\nCPU测试完成!\n")

    # 保存和可视化结果
    save_results_to_csv(gpu_results, cpu_results)
    visualize_results(gpu_results, cpu_results)

    print("\n测试完成!")