import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import loadmat
import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from EEGNet import EEGNet
from InceptionEEG import InceptionEEG
import LoadData as loadData
from collections import namedtuple
from tqdm import tqdm


def load_eeg(filename, event_key='production'):
    EPOCHS = np.load(filename)
    dataset = {}

    # 遍历 npz 文件中的所有数组名
    for key in tqdm(EPOCHS.files):
        if event_key in key:
            dataset[key] = EPOCHS[key]

    return dataset

###============================ Use the GPU to train ============================###
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)

###============================ Setup for saving the model ============================###
current_working_dir = f'../model/'
n_model = int(len(glob.glob('../model/*.pth'))/2 + 1)

###============================ Load data ============================###
filename1 = '../language_average_0.5-80hz_icaLabel95confidence_eyes_40sessions.npz'
filename2 = '../language_average_0.5-80hz_icaLabel95confidence_eyes_20sessions.npz'

dataset1 = load_eeg(filename1, event_key='perception')
dataset2 = load_eeg(filename2, event_key='perception')
EventPerception = {**dataset1, **dataset2}

EventProduction = {}
dataset1 = load_eeg(filename1, event_key='production')
dataset2 = load_eeg(filename2, event_key='production')
EventProduction = {**dataset1, **dataset2}

print(len(EventPerception))
print(len(EventProduction))


select_stim = ['4','9','12','15','28','30']
stim2label = {stim_value: index for index, stim_value in enumerate(select_stim)}

###============================ Manage data ============================###
trial_info = namedtuple('trial_structure', ['subject', 'task', 'stim', 'epoch', 'label'])
eeg_load_trial = namedtuple('eeg_preload_trial', ['data', 'label'])


def process_trials(EventData, select_stim, stim2label, fs=250, times=3.9):
    trials = []  # Trials will be stored here

    for key in tqdm(EventData.keys()):
        subject, task, stim = key.split('_')

        if stim in select_stim:
            num_epoch, channels, _ = EventData[key].shape   # num_epoch, channels, samples

            for epoch_index in range(num_epoch):
                trials.append(
                    trial_info(
                        subject=subject,
                        task=task,
                        stim=stim,
                        epoch=epoch_index,
                        label=stim2label[stim],
                    )
                )

    window = int(fs * times)
    X = np.zeros((len(trials), 1, channels, window))
    Y = np.zeros((len(trials)), dtype=int)

    for idx, single_trial in tqdm(enumerate(trials), total=len(trials)):
        subject = single_trial.subject
        task = single_trial.task
        stim = single_trial.stim
        epoch = single_trial.epoch
        text = f"{subject}_{task}_{stim}"
        X[idx, 0, :, :] = EventData[text][epoch][:, :window]
        Y[idx] = single_trial.label

    return X, Y

# Example usage:
X, Y = process_trials(EventPerception, select_stim, stim2label)
X1, Y1 = process_trials(EventProduction, select_stim, stim2label)


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
test_size       = 0.1
val_size        = 0.1
best_acc        = 0
training_epochs = 80
batch_size      = 50

###============================ Split data & Cross validate ============================###
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0, shuffle=True)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=0, shuffle=True)

# 输出检查
print("Training Set:", X_train.shape, Y_train.shape)
print("Validation Set:", X_val.shape, Y_val.shape)
print("Testing Set:", X_test.shape, Y_test.shape)


# Train
def train_model(model, optimizer, criterion, X_train, Y_train, X_val, Y_val, model_name):
    global n_model
    best_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epo in range(training_epochs):
        running_loss = 0.0
        running_accuracy = 0.0

        x_train = torch.from_numpy(X_train).float()
        x_val = torch.from_numpy(X_val).float()
        y_train = torch.from_numpy(Y_train).long()
        y_val = torch.from_numpy(Y_val).long()

        train_data = loadData.BCICDataLoader(x_train, y_train, batch_size=batch_size)

        model.train()
        for inputs, target in tqdm(train_data):
            inputs = inputs.to(device).requires_grad_()
            target = target.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == target).float().mean()
            running_accuracy += acc / len(train_data)
            running_loss += loss.detach().item() / len(train_data)

        print(f"\nTrain : Epoch : {epo} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
        train_losses.append(running_loss)
        train_accuracies.append(running_accuracy)

        model.eval()
        with torch.no_grad():
            inputs = x_val.to(device)
            val_probs = model(inputs)
            val_acc = (val_probs.argmax(dim=1) == y_val.to(device)).float().mean()

        print(f"Validation: Epoch {epo} - acc: {val_acc:.4f}")
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            PATH = current_working_dir + f'best_{model_name}_model_{n_model}.pth'
            torch.save({
                'model_state_dict': deepcopy(model.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
            }, PATH)

    return train_losses, train_accuracies, val_accuracies, best_acc


###============================ Initialization model ============================###

modelA = EEGNet(kernel_size=64, num_channel=chans, num_class=n_classes, len_window=samples).to(device) # Use EEGNet
optimizerA = torch.optim.Adam(modelA.parameters(), lr=0.001, betas=(0.9, 0.999))
criterionA = nn.CrossEntropyLoss().to(device)

modelB = InceptionEEG(num_channel=chans, num_class=n_classes, len_window=samples).to(device)
optimizerB = torch.optim.Adam(modelB.parameters(), lr=0.001, betas=(0.9, 0.999))
criterionB = nn.CrossEntropyLoss().to(device)

# 训练模型A
print("Training model A: EEGNet")
train_losses_A, train_accuracies_A, val_accuracies_A, best_acc_A = train_model(
    modelA, optimizerA, criterionA, X_train, Y_train, X_val, Y_val, "EEGNet"
)

# 训练模型B
print("Training model B: InceptionEEG")
train_losses_B, train_accuracies_B, val_accuracies_B, best_acc_B = train_model(
    modelB, optimizerB, criterionB, X_train, Y_train, X_val, Y_val, "InceptionEEG"
)

print(f"Best Validation Accuracy for EEGNet: {best_acc_A:.4f}")
print(f"Best Validation Accuracy for InceptionEEG: {best_acc_B:.4f}")

# 将准确度从 tensor 转换为 numpy 数组
train_accuracies_A_cpu = [x.cpu().numpy() for x in train_accuracies_A]
val_accuracies_A_cpu = [x.cpu().numpy() for x in val_accuracies_A]
train_accuracies_B_cpu = [x.cpu().numpy() for x in train_accuracies_B]
val_accuracies_B_cpu = [x.cpu().numpy() for x in val_accuracies_B]

# 绘制准确度和损失曲线
epochs = range(1, training_epochs + 1)

plt.figure(figsize=(12, 6))

# 绘制模型 A 和 B 的训练和验证损失在同一幅图上
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_A, 'b-', label='Training Loss A')
plt.plot(epochs, train_losses_B, 'm-', label='Training Loss B')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制模型 A 和 B 的训练和验证准确度在同一幅图上
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies_A_cpu, color='darkgreen', label='Training Accuracy A')
plt.plot(epochs, val_accuracies_A_cpu, color='lightgreen', label='Validation Accuracy A')
plt.plot(epochs, train_accuracies_B_cpu, color='red', label='Training Accuracy B')
plt.plot(epochs, val_accuracies_B_cpu, color='pink', label='Validation Accuracy B')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

###============================ Test model ============================###
PATH_A = current_working_dir + f'best_EEGNet_model_{n_model}.pth'
PATH_B = current_working_dir + f'best_InceptionEEG_model_{n_model}.pth'

# 加载模型 A 的参数
checkpoint_A = torch.load(PATH_A)
modelA.load_state_dict(checkpoint_A['model_state_dict'], strict=False)
modelA.eval()

# 加载模型 B 的参数
checkpoint_B = torch.load(PATH_B)
modelB.load_state_dict(checkpoint_B['model_state_dict'], strict=False)
modelB.eval()

# 将测试数据转换为 Tensor
x_test = torch.from_numpy(X_test).to(device).float()
y_test = torch.from_numpy(Y_test).to(device).long()

# 测试模型 A
probs_A = modelA(x_test)
acc_A = (probs_A.argmax(dim=1) == y_test).float().mean()
print("Model A - Classification accuracy: %f" % (acc_A))

# 测试模型 B
probs_B = modelB(x_test)
acc_B = (probs_B.argmax(dim=1) == y_test).float().mean()
print("Model B - Classification accuracy: %f" % (acc_B))

# ================== Test Result Visualization ==================
# 获取模型 A 的预测结果和真实标签
predictions_A = probs_A.argmax(dim=1).cpu().numpy()
true_labels_A = y_test.cpu().numpy()

# 生成模型 A 的分类报告
print("Model A - Classification Report:")
print(classification_report(true_labels_A, predictions_A))

# 生成模型 A 的混淆矩阵
conf_matrix_A = confusion_matrix(true_labels_A, predictions_A)

# 可视化模型 A 的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_A, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=select_stim, yticklabels=select_stim)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Model A - Confusion Matrix')
plt.show()

# ================== 模型 B ==================#
# 获取模型 B 的预测结果和真实标签
predictions_B = probs_B.argmax(dim=1).cpu().numpy()
true_labels_B = y_test.cpu().numpy()

# 生成模型 B 的分类报告
print("Model B - Classification Report:")
print(classification_report(true_labels_B, predictions_B))

# 生成模型 B 的混淆矩阵
conf_matrix_B = confusion_matrix(true_labels_B, predictions_B)

# 可视化模型 B 的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_B, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=select_stim, yticklabels=select_stim)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Model B - Confusion Matrix')
plt.show()
