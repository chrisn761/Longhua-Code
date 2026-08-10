import warnings, os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\szx\2026-yolo12\yolov12-seg.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'E:\szx\2026-yolo12\segment-yolo12.yaml',
                cache=False,
                imgsz=960,
                epochs=600,
                batch=4,
                # close_mosaic=0,
                # workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='yolo12-seg-0108seg-2'
                     '',
                )
    # =========================================================
    # 📝 新增功能：自动运行验证并保存控制台表格到 txt
    # =========================================================

    # 获取本次训练结果的保存目录 (例如 runs/train/yolo12-seg-result-table)
    save_dir = model.trainer.save_dir

    # 自动找到刚才训练出的最佳权重 best.pt
    best_weight_path = os.path.join(save_dir, 'weights', 'best.pt')

    # 定义保存表格的文件名
    table_file_path = os.path.join(save_dir, 'final_table_results.txt')

    print(f"\n💾 正在生成详细结果表格，将保存至: {table_file_path} ...")

    # 重新加载最佳权重 (确保验证的是最好的模型，而不是最后一轮的模型)
    best_model = YOLO(best_weight_path)

    # 【关键技巧】将控制台输出“劫持”到文件中
    original_stdout = sys.stdout  # 备份原本的打印通道

    try:
        with open(table_file_path, 'w', encoding='utf-8') as f:
            sys.stdout = f  # 将打印通道指向文件

            print(f"=== YOLOv12-Seg 最终验证结果表 ===")
            print(f"数据源: {data_yaml_path}")
            print(f"权重路径: {best_weight_path}\n")

            # 运行验证 (verbose=True 会强制打印那个表格)
            best_model.val(data=data_yaml_path, split='val', verbose=True)

    finally:
        sys.stdout = original_stdout  # 无论如何，把打印通道恢复回来

    print("✅ 表格保存成功！请去结果文件夹查看 final_table_results.txt")