import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
from architecture.model_standard_EEGNet import EEGNet
import LoadData as loadData
from scipy.io import loadmat
import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

###============================ Use the GPU to train ============================###
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)

###============================ Setup for saving the model ============================###
current_working_dir = f'/content/driver/MyDrive/lhw/model'

###============================ Import Labels ============================###

label = loadmat('./src/label.mat')
label = label['label']

extracted_labels = [item[0] for sublist in label for item in sublist]
print(extracted_labels)

left_labels = np.arange(0,62,2)
print(len(left_labels))

select_stim = ['4','6','9','12','15','28','30']
stim2label = {stim_value: index for index, stim_value in enumerate(select_stim)}


src_files = sorted(glob.glob('./src/Fs250_0.5-80hz_20sessions_perception_*_src.mat'))
label_files = sorted(glob.glob('./src/Fs250_0.5-80hz_20sessions_perception_*_label.npy'))

###============================ Prepare DATA ============================###
X_list = []
Y_list = []

for src_file, label_file in zip(src_files, label_files):
    # 加载label文件
    label = np.load(label_file)
    idx = [index for index, i in enumerate(label) if str(i) in select_stim]
    y = label[idx]
    Y = [stim2label[str(i)] for i in y]


    # 加载src文件
    src_data = loadmat(src_file)
    src = src_data['src']
    X = src[:, left_labels, :]
    X = X[idx, :, :]
    X = np.expand_dims(X, axis=1)  # 增加一个维度

    # 将数据添加到列表中
    X_list.append(X)
    Y_list.append(Y)

    print(len(Y))

# 将列表转换为numpy数组，并按顺序拼接
X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

print(X.shape)
print(Y.shape)

###============================ Initialization parameters ============================###
chans           = X.shape[2]
samples         = X.shape[3]
n_classes       = len(select_stim)
kernelLength    = 64
kernelLength2   = 16
F1              = 8   #################### 4 #########################
D               = 2
F2              = F1 * D
dropoutRate     = 0.5
test_size       = 0.2
val_size       = 0.1
best_acc        = 0

###============================ Split data & Cross validate ============================###
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0, shuffle=True)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=0, shuffle=True)


# 输出检查
print("Training Set:", X_train.shape, Y_train.shape)
print("Validation Set:", X_val.shape, Y_val.shape)
print("Testing Set:", X_test.shape, Y_test.shape)

###============================ Initialization model ============================###
model = EEGNet(kernel_size=64, num_channel=chans, num_class=n_classes, len_window=samples)
# model = InceptionEEG(num_channel=chans, num_class=n_classes, len_window=samples)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss().to(device)

###============================ Train model ============================###

training_epochs = 50
batch_size      = 40

# 记录训练过程中的指标
train_losses = []
train_accuracies = []
val_accuracies = []

for epo in range(0, training_epochs):

    running_loss = 0.0
    running_accuracy = 0.0

    x_train = torch.from_numpy(X_train).float()  # 将数据转换为 Tensor
    x_val = torch.from_numpy(X_val).float()

    y_train = torch.from_numpy(Y_train).long()
    y_val = torch.from_numpy(Y_val).long()

    train_data = loadData.BCICDataLoader(x_train, y_train, batch_size=batch_size)

    model.train()
    for inputs, target in tqdm(train_data):
        inputs = inputs.to(device).requires_grad_()
        target = target.to(device)

        optimizer.zero_grad()  # 重置梯度

        # forward + backward + optimize
        output = model(inputs)  # 通过模型计算输出
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新权重

        # 计算准确度
        acc = (output.argmax(dim=1) == target).float().mean()
        running_accuracy += acc / len(train_data)
        running_loss += loss.detach().item() / len(train_data)

    print(f"\nTrain : Epoch : {epo} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
    train_losses.append(running_loss)
    train_accuracies.append(running_accuracy)

    # 验证
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算以节省内存
        inputs = x_val.to(device)
        val_probs = model(inputs)
        val_acc = (val_probs.argmax(dim=1) == y_val.to(device)).float().mean()  # 使用正确的标签变量

    print(f"Validation: Epoch {epo} - acc: {val_acc:.4f}")
    val_accuracies.append(val_acc)

    # 检查并保存最优模型
    if val_acc > best_acc:
        best_acc = val_acc  # 更新最优准确度
        PATH = current_working_dir + 'best_EEGNet_model.pth'
        # PATH = current_working_dir + 'best_InceptionEEG_model.pth'
        torch.save({
            'model_state_dict': deepcopy(model.state_dict()),
            'optimizer_state_dict': deepcopy(optimizer.state_dict()),
        }, PATH)  # 保存模型状态和优化器状态

print(f"Best Validation Accuracy: {best_acc:.4f}")

###============================ Test model ============================###

x_test = torch.from_numpy(X_test).to(device).float()
y_test = torch.from_numpy(Y_test).to(device).long()

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
probs = model(x_test)
acc = (probs.argmax(dim=1) == y_test).float().mean()
print("Classification accuracy: %f " % (acc))
train_accuracies_cpu = [x.cpu().numpy() for x in train_accuracies]
val_accuracies_cpu = [x.cpu().numpy() for x in val_accuracies]

# 绘制准确度和损失曲线
epochs = range(1, training_epochs + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies_cpu, 'g-', label='Training Accuracy')
plt.plot(epochs, val_accuracies_cpu, 'r-', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

###============================ Train Plot ============================###
train_accuracies_cpu = [x.cpu().numpy() for x in train_accuracies]
val_accuracies_cpu = [x.cpu().numpy() for x in val_accuracies]

# 绘制准确度和损失曲线
epochs = range(1, training_epochs + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies_cpu, 'g-', label='Training Accuracy')
plt.plot(epochs, val_accuracies_cpu, 'r-', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

###============================ Test Plot ============================###

# 获取预测结果和真实标签
predictions = probs.argmax(dim=1).cpu().numpy()
true_labels = y_test.cpu().numpy()

# 生成分类报告
print("Classification Report:")
print(classification_report(true_labels, predictions))

# 生成混淆矩阵
conf_matrix = confusion_matrix(true_labels, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=select_stim, yticklabels=select_stim)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# import os
# import re
# from scipy.io import loadmat
# import numpy as np
# from tqdm import tqdm
#
#
# os.system('cls')
#
# DATASET_FOLDER_PATH = f"F:/Intership/Brainstorm/brainstorm_db/ProductionEpochs/data/Group_analysis/Fs128_1-40hz_20sessions_perception_01_epo2"
#
# sensor_data = {}
# source_data = {}
# for data in os.listdir(DATASET_FOLDER_PATH):
#     if 'data_Epoch_trial' in data:
#         parts = re.split(r'(data_Epoch_trial|\d+)', data)
#         idx = int(parts[3])-1
#         sensor_data[idx] = data
#     elif 'results_atlas_' in data:
#         parts = re.split(r'(results_atlas_240512_|\d+)', data)
#         idx = int(parts[3]) - 1440
#         source_data[idx] = data
#     else:
#         pass
#
# print(len(sensor_data), len(source_data))
#
#
# src = np.zeros((len(source_data), 1, 62, 975))
# for srcIdx in tqdm(len(source_data)):
#     srcPath = os.path.join(DATASET_FOLDER_PATH, source_data[srcIdx])
#     DKT = loadmat(srcPath)
#     src[srcIdx, 0, :, :] = DKT['ImageGridAmp']




