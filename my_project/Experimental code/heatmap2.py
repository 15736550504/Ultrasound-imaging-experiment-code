from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm
import time

# ===== 配置参数 =====
TEST_DIR = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/images/test"  # 测试集目录
MODEL_PATH = "/home/link/Yang/LHL/超声/腹部/my_project/腹部data6.23/yolov8n/weights/best.pt"  # 训练好的模型路径
OUTPUT_DIR = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部热力图/heatmaps Test-n"  # 热力图保存目录

CONFIDENCE = 0.01  # 置信度阈值
IMAGE_SIZE = 640  # YOLOv8的标准尺寸
ALPHA = 0.5  # 热力图透明度

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 打印模型信息
print(f"模型类别数: {model.model.model[-1].nc}")


# 优化的热力图生成函数（只保留热力图）
def generate_heatmap(img_path, output_path):
    """生成并保存热力图图像（只包含热力图）"""
    try:
        # 运行预测
        results = model.predict(img_path, conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)
        result = results[0]

        # 获取原始图像
        orig_img = result.orig_img

        # 检查是否有检测结果
        if result.boxes is None or len(result.boxes) == 0:
            # 没有检测到对象，保存原始图像
            cv2.imwrite(output_path.replace(".jpg", "_nodetection.jpg"), orig_img)
            return False

        # 获取预测结果
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # 创建热力图基础图像
        heatmap_base = np.zeros_like(orig_img[:, :, 0], dtype=np.float32)

        # 为每个检测框添加热力图区域（优化版本）
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box[:4])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = x2 - x1, y2 - y1

            # 创建高斯分布（优化版本）
            sigma = max(width, height) // 4
            if sigma < 1:  # 防止sigma为0
                sigma = 1

            # 创建高斯核
            kernel_size = 6 * sigma + 1
            if kernel_size % 2 == 0:
                kernel_size += 1

            # 生成高斯核
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0 = y0 = kernel_size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # 在热力图上叠加高斯核
            top = max(0, center_y - kernel_size // 2)
            bottom = min(orig_img.shape[0], center_y + kernel_size // 2 + 1)
            left = max(0, center_x - kernel_size // 2)
            right = min(orig_img.shape[1], center_x + kernel_size // 2 + 1)

            # 调整高斯核大小以适应边界
            if top >= bottom or left >= right:
                continue

            g_cropped = g[top - (center_y - kernel_size // 2): bottom - (center_y - kernel_size // 2),
                        left - (center_x - kernel_size // 2): right - (center_x - kernel_size // 2)]

            # 叠加到热力图基础
            heatmap_base[top:bottom, left:right] += g_cropped * conf

        # 归一化热力图
        if heatmap_base.max() > 0:
            heatmap_base = (heatmap_base - heatmap_base.min()) / (heatmap_base.max() - heatmap_base.min() + 1e-8)

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_base), cv2.COLORMAP_JET)

        # 叠加到原始图像（只保留热力图）
        overlay_img = cv2.addWeighted(orig_img, 1 - ALPHA, heatmap_colored, ALPHA, 0)

        # 保存结果（只包含热力图）
        cv2.imwrite(output_path, overlay_img)
        return True

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        # 出错时保存原始图像
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            cv2.imwrite(output_path.replace(".jpg", "_error.jpg"), orig_img)
        return False


# 处理所有图像
def process_all_images():
    """处理目录中的所有图像"""
    img_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"找到 {len(img_files)} 张图像")

    success_count = 0
    for img_name in tqdm(img_files, desc="生成热力图"):
        img_path = os.path.join(TEST_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, f"heatmap_{img_name}")

        if generate_heatmap(img_path, output_path):
            success_count += 1

    return success_count


# 主程序
if __name__ == "__main__":
    print("开始处理所有图像...")
    start_time = time.time()
    success_count = process_all_images()
    elapsed = time.time() - start_time

    # 创建结果摘要
    summary = f"""
    热力图生成结果摘要
    ================================
    模型路径: {MODEL_PATH}
    测试图像目录: {TEST_DIR}
    图像总数: {len(os.listdir(TEST_DIR))}
    成功生成: {success_count}
    失败数量: {len(os.listdir(TEST_DIR)) - success_count}
    输出目录: {OUTPUT_DIR}
    总耗时: {elapsed:.2f}秒
    平均每张图像耗时: {elapsed / len(os.listdir(TEST_DIR)):.2f}秒
    """
    print(summary)

    # 保存摘要到文件
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"处理完成! 结果保存在: {OUTPUT_DIR}")