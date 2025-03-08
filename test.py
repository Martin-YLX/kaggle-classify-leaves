import os
import torch
import pandas as pd
from PIL import Image
from loadData import transform, reverse_label_map
from models.resnet import get_resnet18
from config import TEST_CSV, IMAGES_DIR, DEVICE, SUBMISSION_CSV

# 读取测试数据
df_test = pd.read_csv(TEST_CSV)

# **加载训练好的模型**
num_classes = len(reverse_label_map)
model = get_resnet18(num_classes)
model.load_state_dict(torch.load("resnet18_leaves.pth", map_location=DEVICE, weights_only=True))
model.eval()

# **预测函数**
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(1).item()
    return reverse_label_map[pred_idx]

# **生成 `submission.csv`**
submission = []
for img in df_test["image"]:
    img_path = os.path.join(IMAGES_DIR, img.split("/")[-1])
    pred_label = predict_image(img_path)
    submission.append((img, pred_label))

# **保存 CSV**
submission_df = pd.DataFrame(submission, columns=["image", "label"])
submission_df.to_csv(SUBMISSION_CSV, index=False)
print(f"✅ 提交文件 `{SUBMISSION_CSV}` 生成完成！")