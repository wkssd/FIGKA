import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import glob
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import Logger  # 导入刚刚创建的日志模块
import torch.nn.functional as F

# 获取程序名
program_name = os.path.basename(__file__).replace('.py', '')

# 设置日志文件路径和模型文件路径
log_file, model_file, logfile = Logger.setup_paths(program_name)
print(f"Logging to {log_file}")
print(f"Model will be saved to {model_file}")

# 获取当前时间作为训练开始时间
start_time = time.time()

# 定义超参数
batch_size = 128
learning_rate = 0.001
num_epochs = 60
weight_decay = 1e-5  # L2正则化

Logger.log_only(logfile, f"Training Configuration:\nBatch size: {batch_size}\nLearning rate: {learning_rate}\nNumber of epochs: {num_epochs}\nWeight decay: {weight_decay}")




# 定义数据集类
class GrayscaleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        # 遍历目录，加载图片路径和标签
        for idx, label in enumerate(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                self.label_to_idx[label] = idx
                for image_file in glob.glob(os.path.join(label_path, '*.png')):
                    self.image_paths.append(image_file)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 定义神经网络
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1024 * 1 * 1, 128)  # 1024通道，1x1尺寸的特征图
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.spatial_attention(x) * x  # Apply spatial attention after conv4
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x)))
        x = x.view(-1, 1024 * 1 * 1)  # 调整尺寸以适应全连接层输入
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据
root_dir = '/home/zhangzhan/Dataset/Img_10-30_probability'
dataset = GrayscaleDataset(root_dir=root_dir, transform=transform)

# 显示数据集信息
label_counts = {label: 0 for label in dataset.label_to_idx.keys()}
for label in dataset.labels:
    label_counts[list(dataset.label_to_idx.keys())[label]] += 1

# print("Dataset Information:")
# print(f"Total labels: {len(dataset.label_to_idx)}")
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")
Logger.log_only(logfile, f"\nDataset Information:\nTotal labels: {len(dataset.label_to_idx)}")
for label, count in label_counts.items():
    Logger.log_only(logfile, f"Label {label}: {count} samples")


# 分割数据集, 将数据分成训练集和测试集
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=dataset.labels,
    random_state=42
)

# 显示训练集和测试集的标签分布
train_labels = [dataset.labels[i] for i in train_indices]
test_labels = [dataset.labels[i] for i in test_indices]

train_label_counts = {label: 0 for label in dataset.label_to_idx.keys()}
test_label_counts = {label: 0 for label in dataset.label_to_idx.keys()}

for label in train_labels:
    train_label_counts[list(dataset.label_to_idx.keys())[label]] += 1
for label in test_labels:
    test_label_counts[list(dataset.label_to_idx.keys())[label]] += 1


logging.info("\nTraining set information:")
for label, count in train_label_counts.items():
    logging.info(f"Label {label}: {count} samples")

logging.info("\nTest set information:")
for label, count in test_label_counts.items():
    logging.info(f"Label {label}: {count} samples")
Logger.log_only(logfile, f"\nTraining set information:")
for label, count in train_label_counts.items():
    Logger.log_only(logfile, f"Label {label}: {count} samples")
Logger.log_only(logfile, f"\nTest set information:")
for label, count in test_label_counts.items():
    Logger.log_only(logfile, f"Label {label}: {count} samples")



# 创建数据加载器
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# 模型初始化
num_classes = len(os.listdir(root_dir))
model = SimpleCNN(num_classes=num_classes).to(device)  # 将模型移动到GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型并在每个周期结束后进行测试
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 学习率调度
    scheduler.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")


# 训练结束时间
end_time = time.time()
training_duration = end_time - start_time
training_duration_str = time.strftime("%H:%M:%S", time.gmtime(training_duration))


# 详细测试结果展示
print("\nDetailed test results:")
model.eval()
correct = 0
total = 0
label_correct = {label: 0 for label in dataset.label_to_idx.keys()}
label_total = {label: 0 for label in dataset.label_to_idx.keys()}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(labels)):
            true_label = list(dataset.label_to_idx.keys())[labels[i].item()]
            predicted_label = list(dataset.label_to_idx.keys())[predicted[i].item()]
            if predicted[i].item() != labels[i].item():
                Logger.log_only(logfile, f"Sample {i+1}: True Label: {true_label}, Predicted Label: {predicted_label}, Correct: False")
                # print(f"Sample {i+1}: True Label: {true_label}, Predicted Label: {predicted_label}, Correct: False")
            if predicted[i].item() == labels[i].item():
                label_correct[true_label] += 1
            label_total[true_label] += 1

# 每种标签的分类准确率
print("\nClassification accuracy for each label:")
for label in label_correct.keys():
    accuracy = 100 * label_correct[label] / label_total[label] if label_total[label] > 0 else 0
    print(f"Label {label}: Accuracy: {accuracy:.2f}%")


print(f"\nTraining started at: {time.ctime(start_time)}")
print(f"Training ended at: {time.ctime(end_time)}")
print(f"Total training time: {training_duration_str}")
# Logger.log_only(logfile, f"\nTraining started at: {time.ctime(start_time)}")
# Logger.log_only(logfile, f"Training ended at: {time.ctime(end_time)}")
# Logger.log_only(logfile, f"Total training time: {training_duration:.2f} seconds")

# 保存模型
# torch.save(model.state_dict(), f'/home/zhangzhan/GCN_train_2/Model/{program_name}_{start_time_str}_model.pth')
torch.save(model.state_dict(), model_file)
# Logger.log_only(logfile, f"Model saved to /home/zhangzhan/GCN_train_2/Model/CNNv6_model.pth")

