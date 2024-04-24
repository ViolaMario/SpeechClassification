import numpy as np
from collections import namedtuple, defaultdict
from tqdm import tqdm
import torch.nn as nn
from EEGNet import EEGNet
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


filename = f"F:\\Intership\\Large_Spanish_EEG-main\\data\\npz\\average_1-40hz_icaLabel95confidence_eyes_40sessions.npz"
EPOCHS = np.load(filename)

# 创建一个空字典来存储名字带有'production'的数组
EventProduction = {}

# 遍历npz文件中的所有数组名
for key in EPOCHS.files:
    if 'production' in key:
        EventProduction[key] = EPOCHS[key]

Y = EventProduction[key][0][:, :]

trial_info = namedtuple('trial_structure', ['subject', 'task', 'stim', 'epoch', 'label'])
eeg_load_trial = namedtuple('eeg_preload_trial', ['data', 'label'])

trials = []  # Trials will be stored here
stims = list(range(1, 31))
stim2id = {stim_value: index+1 for index, stim_value in enumerate(stims)}

for key in tqdm(EventProduction.keys()):
    subject, task, stim = key.split('_')
    num_epoch, channels, samples = EventProduction[key].shape # num_epoch, channels, samples

    for epoch_index in range(num_epoch):
        trials.append(
            trial_info(
                subject=subject,
                task=task,
                stim=stim,
                epoch=epoch_index,
                label=stim2id[int(stim)],
            )
        )

fs = 128
window = fs*5   # 5 seconds
X = np.zeros((len(trials), channels, window, 1))
Y = np.zeros((len(trials)))

for idx, single_trial in tqdm(enumerate(trials), total=len(trials)):
    subject = single_trial.subject
    task = single_trial.task
    stim = single_trial.stim
    epoch = single_trial.epoch
    label = single_trial.label
    text = f"{subject}_{task}_{stim}"
    X[idx, :, :, 0] = EventProduction[text][epoch][:, :]
    Y[idx] = label

X_train = np.transpose(X, (0, 3, 1, 2))
Y_train = Y
print(X_train.shape)

# 创建 TensorDataset 和 DataLoader
dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # 设定批量大小

# Training
# 实例化模型
model = EEGNet(n_classes=30, channels=64, samples=640)

# 选择交叉熵损失，因为它适用于多分类任务
criterion = nn.CrossEntropyLoss()

# 使用 Adam 作为优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 可以根据需要调整学习率

# 训练循环
num_epochs = 10  # 训练 10 个 epoch
model.train()  # 将模型设置为训练模式

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        # 将梯度置零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


# 创建验证集 DataLoader
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# # 计算准确率
# accuracy = calculate_accuracy(model, val_loader)
# print("Validation Accuracy:", accuracy)
#
# # 打印混淆矩阵
# plot_confusion_matrix(model, val_loader)
#
# # 打印分类报告
# print_classification_report(model, val_loader)