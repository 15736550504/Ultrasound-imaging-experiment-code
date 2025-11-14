from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import glob
import os

if __name__ == '__main__':
    # 加载训练好的模型（需替换为你的模型路径）
    model = YOLO("/home/link/Yang/LHL/超声/腹部/my_project/腹部data6.23/yolov8n-UFEA best训练 b32/weights/epoch70.pt")  # 示例路径："肺水肿5.9.2/yolov8n1/weights/best.pt"
    # 定义新的17个类别名称
    new_class_names = [
        "Fatty liver",
        "gallstones",
        "the gallbladder polyps",
        "liver hemangioma inside",
        "liver cyst",
        "intrahepatic calcification",
        "placeholder in the liver",
        "kidney stones",
        "renal cysts",
        "renal placeholder",
        "spleen placeholder",
        "bladder stones",
        "the gallbladder placeholder",
        "splenic hemangioma",
        "spleen cyst",
        "bladder placeholder",
        "placeholder pancreas"
    ]

    # 替代方案：直接修改模型内部属性
    try:
        # 尝试直接修改模型内部属性
        model.model.names = new_class_names
    except AttributeError:
        # 如果不行，尝试修改配置文件中的类名
        model.model.yaml["names"] = new_class_names
        model.model.names = new_class_names
    # 待预测图片目录（需替换为你的图片路径）
    image_dir = "/home/link/Yang/LHL/超声/腹部/补充数据训练/肺水肿data5.9.2/predict"
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))[:3]  # 获取前3张图片
    # 创建对比画布（3行2列布局）
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, img_path in enumerate(image_paths):
        # 读取并转换原始图片
        original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # 进行预测（调整conf参数可修改置信度阈值）
        results = model.predict(img_path, conf=0.5)
        annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        # 绘制原图
        axes[i, 0].imshow(original_img)
        axes[i, 0].axis("off")
        # axes[i, 0].set_title(f"Original {i + 1}", fontsize=12)
        # 绘制预测结果
        axes[i, 1].imshow(annotated_img)
        axes[i, 1].axis("off")
        # axes[i, 1].set_title(f"Predicted {i + 1}", fontsize=12)
    # 保存并显示结果
    output_path = os.path.join("/home/link/Yang/LHL/超声/腹部/补充数据训练/肺水肿data5.9.2", "predictions_comparison.jpg")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"预测对比图已保存至：{output_path}")