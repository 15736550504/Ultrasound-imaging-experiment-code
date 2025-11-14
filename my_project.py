# -*- coding: utf-8 -*-
from torch.optim.adamw import AdamW

from ultralytics import YOLO
if __name__ == '__main__':
    yaml = r"model/yolov8n"
    data = r"/home/link/Yang/LHL/超声/腹部/补充数据训练/腹部data6.23/dataset.yaml"
    project = "腹部data6.23"
    name = "yolov8n"
    path = project + "/" + name
    model = YOLO(yaml, task="detect")
    model.train(data=data, project=project, name=name, imgsz=640, batch=16,patience=30,
                epochs=300, device=0, exist_ok=True,cos_lr=True,save_period=10
                )

    model = YOLO(path + "/best.pt")
    model.val(data=data, project=path,name="test", split="test", plots=True, exist_ok=True, device=0)

    # model=YOLO(r"best.pt")
    # model.predict(r"liver.avi",show=True,save=False,conf=0.01,iou=0.1,save_txt=True,device=0)
