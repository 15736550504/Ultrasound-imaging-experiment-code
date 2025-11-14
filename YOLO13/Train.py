from ultralytics import YOLO
import torch
torch.use_deterministic_algorithms(False)  # 关闭确定性计算
torch.backends.cudnn.deterministic = False  # 关闭 cuDNN 确定性
if __name__ == '__main__':

    yaml = r"yolov10n.yaml"
    data = r"/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/dataset.yaml"
    project = "腹部data6.23"
    name = "yolov10n模型"
    path = project + "/" + name
    model = YOLO(yaml, task="detect")
    # model.train(data=data, project=project, name=name, imgsz=640, batch=16,patience=30,
    #                 epochs=300, device=0, exist_ok=True,cos_lr=True,save_period=10
    #             )

    model = YOLO(path + "/weights/best.pt")
    model.val(data=data, batch=1,project=path, name="test", split="test", plots=True, exist_ok=True, device=0)