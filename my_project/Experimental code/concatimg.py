import os
from PIL import Image


def combine_images(folder1, folder2, folder3, output_folder):
    # 获取三个文件夹中的图片文件列表（按字母顺序排序）
    img_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    files1 = sorted([f for f in os.listdir(folder1)
                     if os.path.splitext(f)[1].lower() in img_ext])
    files2 = sorted([f for f in os.listdir(folder2)
                     if os.path.splitext(f)[1].lower() in img_ext])
    files3 = sorted([f for f in os.listdir(folder3)
                     if os.path.splitext(f)[1].lower() in img_ext])

    # 检查文件数量是否相同
    if len(files1) != len(files2) or len(files2) != len(files3):
        missing1 = set(files1) - set(files2) - set(files3)
        missing2 = set(files2) - set(files1) - set(files3)
        missing3 = set(files3) - set(files1) - set(files2)
        error_msg = "图片数量不匹配:\n"
        error_msg += f"文件夹1: {len(files1)}张\n文件夹2: {len(files2)}张\n文件夹3: {len(files3)}张"
        if missing1: error_msg += f"\n只在文件夹1中的文件: {', '.join(missing1)[:50]}..."
        if missing2: error_msg += f"\n只在文件夹2中的文件: {', '.join(missing2)[:50]}..."
        if missing3: error_msg += f"\n只在文件夹3中的文件: {', '.join(missing3)[:50]}..."
        raise ValueError(error_msg)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    for i, filename in enumerate(files1):
        # 构建文件路径
        img_path1 = os.path.join(folder1, filename)
        img_path2 = os.path.join(folder2, files2[i])
        img_path3 = os.path.join(folder3, files3[i])

        try:
            # 打开三张图片
            images = [Image.open(img_path1), Image.open(img_path2), Image.open(img_path3)]

            # 处理图片模式（统一为RGB，解决透明通道问题）
            processed_images = []
            for img in images:
                if img.mode == 'RGBA':
                    # 创建白色背景来处理透明通道
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
                    processed_images.append(background)
                else:
                    processed_images.append(img.convert('RGB'))

            # 计算拼接后图片的尺寸
            width = sum(img.width for img in processed_images)
            height = max(img.height for img in processed_images)

            # 创建新画布（使用第一个图片的颜色模式）
            combined = Image.new('RGB', (width, height))

            # 拼接图片
            x_offset = 0
            for img in processed_images:
                # 垂直居中对齐
                y_offset = (height - img.height) // 2
                combined.paste(img, (x_offset, y_offset))
                x_offset += img.width

            # 保存结果（使用第一个文件夹的文件名）
            output_path = os.path.join(output_folder, filename)
            # 保持原始格式（通过扩展名判断）
            if filename.lower().endswith(('.png', '.gif', '.bmp', '.tiff', '.webp')):
                combined.save(output_path)
            else:
                combined.save(output_path, quality=95)  # JPG使用高质量

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            continue


if __name__ == "__main__":
    # 文件夹路径设置（替换为实际路径）
    folder1 = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/annotated_test"  # 主文件夹，文件名由此决定
    folder2 = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部热力图/annotated_imagesprodict-n"
    folder3 = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部热力图/annotated_imagesprodict"
    output_folder = "/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部热力图/heatmaps concat2"

    try:
        combine_images(folder1, folder2, folder3, output_folder)
        print(f"图片拼接完成！共处理 {len(os.listdir(folder1))} 张图片。")
    except Exception as e:
        print(f"处理失败: {str(e)}")
