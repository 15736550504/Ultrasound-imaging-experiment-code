import cv2
import os
import yaml

# 配置路径
dataset_path = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23"  # 替换为你的数据集路径
images_dir = os.path.join(dataset_path, "images/test")
labels_dir = os.path.join(dataset_path, "labels/test")
output_dir = os.path.join(dataset_path, "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/annotated_test")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取类别名称（需要data.yaml文件）
with open(os.path.join(dataset_path, "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/dataset.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
class_names = data_config["names"]

# 颜色映射（不同类别不同颜色）
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # 可自定义

# 遍历测试集图片
for img_file in os.listdir(images_dir):
    # 读取图片
    img_path = os.path.join(images_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # 获取对应标签文件
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    # 读取标签内容
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.readlines()

        # 绘制每个标注框
        for label in labels:
            cls_id, x_center, y_center, bw, bh = map(float, label.split())
            cls_id = int(cls_id)

            # 转换坐标（YOLO格式 -> 像素坐标）
            x_center *= w
            y_center *= h
            bw *= w
            bh *= h

            # 计算边界框坐标
            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)

            # 绘制边界框
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 添加类别标签
            label_text = f"{class_names[cls_id]}"
            cv2.putText(img, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 保存标注后的图片
    output_path = os.path.join(output_dir, img_file)
    cv2.imwrite(output_path, img)

print(f"标注完成！结果保存在：{output_dir}")