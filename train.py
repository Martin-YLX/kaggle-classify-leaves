import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loadData import LeavesDataset, transform, df_train, label_map
from models.resnet import get_resnet18
from config import BATCH_SIZE, EPOCHS, LR, DEVICE

# **数据加载**
train_dataset = LeavesDataset(df_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# **加载模型**
num_classes = len(label_map)
model = get_resnet18(num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# **训练循环**
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {total_correct/len(train_dataset):.4f}")

# **保存模型**
torch.save(model.state_dict(), "resnet18_leaves.pth")
print("✅ 训练完成，模型已保存 resnet18_leaves.pth！")