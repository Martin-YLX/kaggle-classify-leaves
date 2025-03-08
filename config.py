import os
import torch

# **数据路径**
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
SUBMISSION_CSV = "submission.csv"  # 预测结果保存路径

# **训练参数**
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ 运行设备: {DEVICE}")