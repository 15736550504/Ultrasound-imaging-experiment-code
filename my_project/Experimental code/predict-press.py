import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO
import shutil
import tempfile
import copy
import json
import re


def apply_image_transformations(img, brightness=1.0, contrast=1.0, saturation=1.0,
                                noise_level=0.0, scale_factor=1.0):
    """
    应用亮度、对比度、饱和度、噪声和分辨率变换到图像
    """
    # 首先应用分辨率变换
    if scale_factor != 1.0:
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)

        # 确保尺寸至少为1像素
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # 下采样
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 上采样回原始尺寸
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 应用亮度、对比度和饱和度变换
    img = img.astype(np.float32) / 255.0
    img = img * brightness
    img = np.clip(img, 0, 1)
    mean = np.mean(img, axis=(0, 1))
    img = (img - mean) * contrast + mean
    img = np.clip(img, 0, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = (img * 255).astype(np.uint8)

    # 应用噪声
    if noise_level > 0:
        # 生成高斯噪声
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.int16)
        img = img.astype(np.int16) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def resolve_path(base_path, relative_path):
    """解析相对路径为绝对路径"""
    if os.path.isabs(relative_path):
        return relative_path

    config_dir = os.path.dirname(base_path)
    abs_path = os.path.join(config_dir, relative_path)
    if os.path.exists(abs_path):
        return abs_path

    if os.path.exists(relative_path):
        return relative_path

    return relative_path


def create_correct_directory_structure(base_dir, test_images, labels_path):
    """创建符合 Ultralytics 要求的目录结构"""
    # 创建标准目录结构
    images_dir = base_dir / "images" / "test"
    labels_dir = base_dir / "labels" / "test"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 复制图像文件
    for img_path in test_images:
        dest_path = images_dir / img_path.name
        shutil.copy(str(img_path), str(dest_path))

    # 复制标签文件 - 使用正确的标签路径
    for img_path in test_images:
        label_name = img_path.stem + ".txt"
        # 从标签目录直接获取标签文件
        src_label = Path(labels_path) / label_name

        if src_label.exists():
            dest_label = labels_dir / label_name
            shutil.copy(str(src_label), str(dest_label))
        else:
            # 不再创建空标签文件
            print(f"警告: 标签文件不存在: {src_label}")

    return images_dir, labels_dir


def extract_class_ap_values(metrics, class_names):
    """从评估结果中提取每个类别的AP值"""
    class_ap50 = {}
    class_ap = {}

    # 打印调试信息
    print("开始提取类别AP值...")

    # 首先尝试从metrics.box中获取
    if hasattr(metrics, 'box'):
        box_metrics = metrics.box

        # 检查是否有ap_class_index属性
        if hasattr(box_metrics, 'ap_class_index'):
            ap_class_index = box_metrics.ap_class_index
            print(f"ap_class_index: {ap_class_index}")

            # 检查是否有ap50属性
            if hasattr(box_metrics, 'ap50'):
                ap50_per_class = box_metrics.ap50
                print(f"ap50_per_class: {ap50_per_class}")

                # 检查是否有ap属性
                if hasattr(box_metrics, 'ap'):
                    ap_per_class = box_metrics.ap
                    print(f"ap_per_class: {ap_per_class}")

                    # 将AP值映射到类别名称
                    for i, class_idx in enumerate(ap_class_index):
                        if class_idx < len(class_names):
                            class_name = class_names[class_idx]
                            if i < len(ap50_per_class):
                                class_ap50[class_name] = ap50_per_class[i]
                            if i < len(ap_per_class):
                                class_ap[class_name] = ap_per_class[i]

    # 如果从box属性无法获取，尝试从results_dict中获取
    if not class_ap50 and hasattr(metrics, 'results_dict'):
        results_dict = metrics.results_dict
        print("可用的结果键:", list(results_dict.keys()))

        # 查找所有包含类别AP值的键
        for key in results_dict.keys():
            # 查找AP50值
            if 'ap50' in key.lower() and 'class' in key.lower():
                # 提取类别索引
                match = re.search(r'class_(\d+)', key)
                if match:
                    class_idx = int(match.group(1))
                    if class_idx < len(class_names):
                        class_name = class_names[class_idx]
                        class_ap50[class_name] = results_dict[key]

            # 查找AP值 (不包含ap50)
            elif 'ap(' in key.lower() and 'class' in key.lower() and 'ap50' not in key.lower():
                # 提取类别索引
                match = re.search(r'class_(\d+)', key)
                if match:
                    class_idx = int(match.group(1))
                    if class_idx < len(class_names):
                        class_name = class_names[class_idx]
                        class_ap[class_name] = results_dict[key]

    # 确保所有类别都有值
    for class_name in class_names:
        if class_name not in class_ap50:
            class_ap50[class_name] = 0.0
        if class_name not in class_ap:
            class_ap[class_name] = 0.0

    print(f"提取的class_ap50: {class_ap50}")
    print(f"提取的class_ap: {class_ap}")

    return class_ap50, class_ap


def evaluate_model_with_transformations(model, dataset_path, transformations, iou_thresh=0.7, device="0"):
    """在应用不同图像变换的情况下评估模型性能"""
    results = []

    # 加载数据配置
    with open(dataset_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # 解析并获取测试集路径
    test_path = resolve_path(dataset_path, data_config.get('test', ''))
    if not test_path or not os.path.exists(test_path):
        print(f"错误: 测试集路径不存在: {test_path}")
        return results

    # 获取原始测试集图像列表
    test_images = list(Path(test_path).glob("*.*"))
    test_images = [img for img in test_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    if not test_images:
        print(f"错误: 测试集中没有找到图像文件: {test_path}")
        return results

    print(f"找到 {len(test_images)} 张测试图像")

    # 修复: 直接使用测试集对应的标签路径
    labels_path = test_path.replace("images/test", "labels/test")
    if not os.path.exists(labels_path):
        # 尝试其他可能的路径
        labels_path = test_path.replace("images", "labels")
        if not os.path.exists(labels_path):
            print(f"错误: 标签目录不存在: {labels_path}")
            return results

    print(f"使用标签目录: {labels_path}")

    # 检查标签文件数量
    label_files = list(Path(labels_path).glob("*.txt"))
    print(f"找到 {len(label_files)} 个标签文件")

    # 创建临时目录用于存储变换后的数据集
    temp_root = Path(tempfile.mkdtemp(prefix="stress_test_"))
    print(f"创建临时目录: {temp_root}")

    # 创建原始数据集的标准结构（用于参考）
    base_images_dir, base_labels_dir = create_correct_directory_structure(
        temp_root / "base",
        test_images,
        labels_path
    )

    # 创建基础数据配置文件
    base_data_config = copy.deepcopy(data_config)
    base_data_config['path'] = str(temp_root / "base")
    base_data_config['test'] = "images/test"
    base_data_config['val'] = "images/test"  # 对于测试使用相同的路径

    base_config_path = temp_root / "base" / "data.yaml"
    with open(base_config_path, 'w') as f:
        yaml.dump(base_data_config, f)

    # 获取类别名称
    class_names = data_config.get('names', [])

    # 如果类别名称是字典，转换为列表
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    print(f"类别名称列表: {class_names}")

    for transform in tqdm(transformations, desc="压力测试进度"):
        # 为当前变换创建目录
        transform_name = transform["name"]
        transform_dir = temp_root / transform_name
        transform_dir.mkdir(parents=True, exist_ok=True)

        # 创建变换后的数据集的标准结构
        trans_images_dir, trans_labels_dir = create_correct_directory_structure(
            transform_dir,
            test_images,
            labels_path
        )

        # 应用变换到所有图像
        for img_path in tqdm(test_images, desc=f"应用变换: {transform_name}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # 应用变换
            transformed_img = apply_image_transformations(
                img,
                brightness=transform.get("brightness", 1.0),
                contrast=transform.get("contrast", 1.0),
                saturation=transform.get("saturation", 1.0),
                noise_level=transform.get("noise_level", 0.0),
                scale_factor=transform.get("scale_factor", 1.0)
            )

            # 保存变换后的图像
            dest_path = trans_images_dir / img_path.name
            cv2.imwrite(str(dest_path), transformed_img)

        # 创建变换后的数据配置文件
        trans_data_config = copy.deepcopy(base_data_config)
        trans_data_config['path'] = str(transform_dir)
        trans_data_config['test'] = "images/test"
        trans_data_config['val'] = "images/test"

        trans_config_path = transform_dir / "data.yaml"
        with open(trans_config_path, 'w') as f:
            yaml.dump(trans_data_config, f)

        # 在变换后的数据集上评估模型
        try:
            metrics = model.val(
                data=str(trans_config_path),
                split="test",
                plots=False,
                device=device,
                iou=iou_thresh,
                verbose=True  # 设置为True以获取更多信息
            )

            # 获取每个类别的AP值
            class_ap50, class_ap = extract_class_ap_values(metrics, class_names)

            # 保存结果
            result = {
                "name": transform_name,
                "metrics": {
                    "map50": getattr(getattr(metrics, 'box', None), 'map50', 0.0),
                    "map": getattr(getattr(metrics, 'box', None), 'map', 0.0),
                    "precision": getattr(getattr(metrics, 'box', None), 'precision', 0.0),
                    "recall": getattr(getattr(metrics, 'box', None), 'recall', 0.0)
                },
                "class_ap50": class_ap50,
                "class_ap": class_ap
            }

            # 如果无法从box属性获取指标，尝试从results_dict获取
            if result["metrics"]["map50"] == 0.0 and hasattr(metrics, 'results_dict'):
                results_dict = metrics.results_dict
                result["metrics"]["map50"] = results_dict.get('metrics/mAP50(B)', 0.0)
                result["metrics"]["map"] = results_dict.get('metrics/mAP50-95(B)', 0.0)
                result["metrics"]["precision"] = results_dict.get('metrics/precision(B)', 0.0)
                result["metrics"]["recall"] = results_dict.get('metrics/recall(B)', 0.0)

            # 添加变换参数
            if "brightness" in transform:
                result["brightness"] = transform["brightness"]
            if "contrast" in transform:
                result["contrast"] = transform["contrast"]
            if "saturation" in transform:
                result["saturation"] = transform["saturation"]
            if "noise_level" in transform:
                result["noise_level"] = transform["noise_level"]
            if "scale_factor" in transform:
                result["scale_factor"] = transform["scale_factor"]

            results.append(result)

        except Exception as e:
            print(f"评估失败: {e}")
            # 创建默认结果
            result = {
                "name": transform_name,
                "metrics": {
                    "map50": 0.0,
                    "map": 0.0,
                    "precision": 0.0,
                    "recall": 0.0
                },
                "class_ap50": {name: 0.0 for name in class_names},
                "class_ap": {name: 0.0 for name in class_names}
            }

            # 添加变换参数
            if "brightness" in transform:
                result["brightness"] = transform["brightness"]
            if "contrast" in transform:
                result["contrast"] = transform["contrast"]
            if "saturation" in transform:
                result["saturation"] = transform["saturation"]
            if "noise_level" in transform:
                result["noise_level"] = transform["noise_level"]
            if "scale_factor" in transform:
                result["scale_factor"] = transform["scale_factor"]

            results.append(result)

    # 清理临时目录
    try:
        shutil.rmtree(temp_root)
        print(f"已清理临时目录: {temp_root}")
    except Exception as e:
        print(f"清理临时目录失败: {e}")

    return results


def save_results(results, output_dir, class_names):
    """保存测试结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 确保 class_names 不为空
    if not class_names and results:
        print("警告: class_names 为空，尝试从结果中提取类别名称")
        if 'class_ap50' in results[0]:
            class_names = list(results[0]['class_ap50'].keys())
        else:
            print("错误: 无法确定类别名称，使用默认占位符")
            class_names = [f"class_{i}" for i in range(18)]  # 默认占位符

    print(f"保存结果，类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")

    # 主结果文件
    with open(os.path.join(output_dir, 'stress_test_results4.csv'), 'w', encoding='utf-8') as f:
        # 写入标题行
        header = "name,brightness,contrast,saturation,noise_level,scale_factor,map50,map,precision,recall"

        # 添加每个类别的AP50和AP列
        for class_name in class_names:
            header += f",{class_name}_ap50,{class_name}_ap"

        f.write(header + "\n")

        for r in results:
            # 写入变换名称
            f.write(f"{r['name']},")

            # 写入变换参数（如果存在）
            f.write(f"{r.get('brightness', '')},")
            f.write(f"{r.get('contrast', '')},")
            f.write(f"{r.get('saturation', '')},")
            f.write(f"{r.get('noise_level', '')},")
            f.write(f"{r.get('scale_factor', '')},")

            # 写入性能指标
            f.write(f"{r['metrics']['map50']},")
            f.write(f"{r['metrics']['map']},")
            f.write(f"{r['metrics']['precision']},")
            f.write(f"{r['metrics']['recall']},")

            # 写入每个类别的AP值
            for i, class_name in enumerate(class_names):
                ap50 = r['class_ap50'].get(class_name, 0.0)
                ap = r['class_ap'].get(class_name, 0.0)
                f.write(f"{ap50},{ap}")
                # 如果不是最后一个类别，添加逗号
                if i < len(class_names) - 1:
                    f.write(",")

            f.write("\n")

    # 保存详细的JSON结果
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    # 配置
    model_path = r"/home/link/Yang/LHL/超声/腹部/my_project/腹部data6.23/yolov8n-UFEA best训练 b32/weights/epoch70.pt"
    data_config = r"/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/dataset.yaml"
    output_dir = r"/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部热力图/stress_test_results"
    device = "0"  # 使用GPU 0

    # 加载模型
    model = YOLO(model_path)

    # 检查数据集配置文件
    if not os.path.exists(data_config):
        print(f"错误: 数据集配置文件不存在: {data_config}")
        return

    # 打印数据集配置信息
    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)
        print("数据集配置:")
        print(f"训练集: {data.get('train', '未指定')}")
        print(f"验证集: {data.get('val', '未指定')}")
        print(f"测试集: {data.get('test', '未指定')}")
        print(f"类别数: {data.get('nc', '未指定')}")
        print(f"类别名称: {data.get('names', '未指定')}")

        test_path = resolve_path(data_config, data.get('test', ''))
        print(f"解析后的测试集路径: {test_path}")
        if test_path and os.path.exists(test_path):
            test_images = list(Path(test_path).glob("*.*"))
            test_images = [img for img in test_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            print(f"测试集存在，包含 {len(test_images)} 张图像")
        else:
            print("错误: 测试集路径不存在")

        # 获取正确的标签目录
        labels_path = test_path.replace("images/test", "labels/test")
        if not os.path.exists(labels_path):
            labels_path = test_path.replace("images", "labels")
            if not os.path.exists(labels_path):
                print(f"错误: 标签目录不存在: {labels_path}")
                return
            else:
                print(f"推导的标签目录: {labels_path}")
        else:
            print(f"配置的标签目录: {labels_path}")

        # 检查标签文件数量
        label_files = list(Path(labels_path).glob("*.txt"))
        print(f"找到 {len(label_files)} 个标签文件")
        if len(label_files) == 0:
            print("错误: 标签目录中没有找到任何标签文件")
            return
        elif len(label_files) < len(test_images):
            print(f"警告: 标签文件数量({len(label_files)})少于测试图像数量({len(test_images)})")

    # 获取类别名称
    class_names = data.get('names', [])

    # 如果类别名称是字典，转换为列表
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    print(f"类别名称: {class_names}")

    # 创建独立参数测试列表
    transformations = []

    # # 1. 测试亮度变化 (对比度和饱和度固定为1.0)
    # for brightness in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,4.5,5.0]:
    #     transformations.append({
    #         "name": f"brightness_{brightness}",
    #         "brightness": brightness,
    #         "contrast": 1.0,
    #         "saturation": 1.0
    #     })

    # 2. 测试对比度变化 (亮度和饱和度固定为1.0)
    for contrast in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,4.5,5.0]:
        transformations.append({
            "name": f"contrast_{contrast}",
            "brightness": 1.0,
            "contrast": contrast,
            "saturation": 1.0
        })

    # # 3. 测试饱和度变化 (亮度和对比度固定为1.0)
    # for saturation in [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 3.3,3.7]:
    #     transformations.append({
    #         "name": f"saturation_{saturation}",
    #         "brightness": 1.0,
    #         "contrast": 1.0,
    #         "saturation": saturation
    #     })
    #
    # # 4. 测试噪声水平 (其他参数固定为1.0/0)高斯噪声
    # for noise_level in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    #     transformations.append({
    #         "name": f"noise_{noise_level",
    #         "noise_level": noise_level
    #     })
    #
    # # 5. 测试分辨率变化
    # for scale_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     transformations.append({
    #         "name": f"resolution_{scale_factor}",
    #         "scale_factor": scale_factor
    #     })

    print(f"将测试 {len(transformations)} 种独立参数变化情况")

    # 清空GPU缓存
    torch.cuda.empty_cache()
    print(
        f"当前GPU内存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 运行压力测试
    results = evaluate_model_with_transformations(
        model,
        data_config,
        transformations,
        iou_thresh=0.7,
        device=device
    )

    # 保存结果
    save_results(results, output_dir, class_names)

    print(f"压力测试完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()