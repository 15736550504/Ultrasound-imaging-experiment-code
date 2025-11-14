from ultralytics import YOLO

# 加载部分训练的模型
model = YOLO('/home/link/Yang/LHL/超声/腹部/YOLO13/腹部data6.23/yolov11n模型/weights/last.pt')

# 恢复训练
results = model.train(resume=True)