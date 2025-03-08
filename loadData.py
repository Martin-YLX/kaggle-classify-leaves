import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import IMAGES_DIR, TRAIN_CSV

# 读取训练数据
df_train = pd.read_csv(TRAIN_CSV)

# 创建类别映射
label_map = {label: idx for idx, label in enumerate(df_train['label'].unique())}
reverse_label_map = {idx: label for label, idx in label_map.items()}
df_train['label_idx'] = df_train['label'].map(label_map)

# 叶子分类数据集
class LeavesDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGES_DIR, self.dataframe.iloc[idx, 0].split("/")[-1])
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx, 2]  # 类别索引
        if self.transform:
            image = self.transform(image)
        return image, label

# **数据增强**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])