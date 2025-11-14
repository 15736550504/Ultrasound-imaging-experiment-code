import time
import numpy as np
from ultralytics import YOLO
import torch


def benchmark_model(model, data, device_type, warmup=10, runs=100):
    """
    基准测试函数，测量模型在指定设备上的推理性能

    参数:
        model: YOLO模型
        data: 数据集配置文件路径
        device_type: 设备类型 ('cuda' 或 'cpu')
        warmup: 预热迭代次数
        runs: 正式测试迭代次数

    返回:
        fps_stats: 包含FPS统计信息的字典
    """
    # 设置设备
    device = torch.device(device_type)
    model.model.to(device)

    # 获取测试数据集
    from ultralytics.data.utils import check_det_dataset
    dataset = check_det_dataset(data)
    test_path = dataset.get('test', dataset.get('val', dataset.get('train')))

    # 加载几张测试图像进行基准测试
    from ultralytics.data.build import load_inference_source
    source = load_inference_source(test_path)

    # 预热
    print(f"正在进行 {warmup} 次预热迭代...")
    for i in range(warmup):
        if i < len(source):
            img = source[i] if isinstance(source, list) else next(source)
            if isinstance(img, str):
                img = np.array(Image.open(img))
            model(img, verbose=False, device=device_type)

    # 正式测试
    print(f"正在进行 {runs} 次性能测试迭代...")
    timings = []
    for i in range(runs):
        if i < len(source):
            img = source[i] if isinstance(source, list) else next(source)
            if isinstance(img, str):
                img = np.array(Image.open(img))

            start_time = time.time()
            model(img, verbose=False, device=device_type)
            end_time = time.time()

            timings.append(end_time - start_time)

    # 计算FPS
    fps_values = [1 / t for t in timings]
    avg_fps = np.mean(fps_values)
    max_fps = np.max(fps_values)
    min_fps = np.min(fps_values)
    std_fps = np.std(fps_values)

    # 计算稳定FPS (去掉最高和最低的10%)
    sorted_fps = sorted(fps_values)
    trim_count = int(len(sorted_fps) * 0.1)
    trimmed_fps = sorted_fps[trim_count:-trim_count] if trim_count > 0 else sorted_fps
    stable_fps = np.mean(trimmed_fps)

    return {
        'device': device_type,
        'avg_fps': avg_fps,
        'max_fps': max_fps,
        'min_fps': min_fps,
        'std_fps': std_fps,
        'stable_fps': stable_fps,
        'runs': runs
    }


if __name__ == '__main__':
    yaml = r"model/yolov8n-ResCRS.yaml"
    data = r"/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/dataset.yaml"
    project = "腹部data6.23"
    name = "yolov8n-UFEA best训练 b32"
    path = project + "/" + name

    # 加载模型
    model = YOLO(yaml, task="detect")
    model = YOLO(path + "/weights/epoch70.pt")

    # 首先进行标准验证
    print("正在进行标准验证...")
    model.val(data=data, project=path, name="test", split="test", plots=True, exist_ok=True, device=0)

    # 测试GPU性能
    print("\n测试GPU性能...")
    gpu_stats = benchmark_model(model, data, 'cuda')

    # 测试CPU性能
    print("\n测试CPU性能...")
    cpu_stats = benchmark_model(model, data, 'cpu')

    # 打印结果
    print("\n" + "=" * 50)
    print("性能测试结果")
    print("=" * 50)

    print(f"\nGPU性能 ({gpu_stats['runs']}次运行):")
    print(f"  平均FPS: {gpu_stats['avg_fps']:.2f}")
    print(f"  最高FPS: {gpu_stats['max_fps']:.2f}")
    print(f"  最低FPS: {gpu_stats['min_fps']:.2f}")
    print(f"  稳定FPS: {gpu_stats['stable_fps']:.2f} (去掉最高和最低10%)")
    print(f"  标准差: {gpu_stats['std_fps']:.2f}")

    print(f"\nCPU性能 ({cpu_stats['runs']}次运行):")
    print(f"  平均FPS: {cpu_stats['avg_fps']:.2f}")
    print(f"  最高FPS: {cpu_stats['max_fps']:.2f}")
    print(f"  最低FPS: {cpu_stats['min_fps']:.2f}")
    print(f"  稳定FPS: {cpu_stats['stable_fps']:.2f} (去掉最高和最低10%)")
    print(f"  标准差: {cpu_stats['std_fps']:.2f}")

    # 计算加速比
    speedup = gpu_stats['avg_fps'] / cpu_stats['avg_fps']
    print(f"\nGPU相对于CPU的加速比: {speedup:.2f}x")