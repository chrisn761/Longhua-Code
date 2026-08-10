import os
from ultralytics import YOLO

# 解决 Windows 下常见的 OpenMP 库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 🌟 终极修改：用 yaml 搭建 Pose 模型结构，并加载 detect 权重作为预训练主干（这叫权重迁移）
    model = YOLO('./net_yaml/yolo12l_CBAM.yaml').load('yolo12l.pt')

    results = model.train(
        data=r'D:\wzy\ultralytics-main\yolo11\data_yaml\HumanBack.yaml', #数据集地址修改了吗，项目名称修改了吗
        epochs=150,
        batch=-1,
        resume=False,
        project='YOLO12_Pose',
        name='2026_back_L-3.18',
        device='0'
    )