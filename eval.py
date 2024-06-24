from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch


class Eval:
    def __init__(self):
        super(eval, self).__init__()

    def calculate_accuracy(model, dataloader):
        correct = 0
        total = 0

        # 设置模型为评估模式
        model.eval()

        with torch.no_grad():
            for data, targets in dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)  # 获取预测值
                total += targets.size(0)
                correct += (predicted == targets).sum().item()  # 计算正确预测数量

        accuracy = correct / total
        return accuracy

    def plot_confusion_matrix(model, dataloader):
        all_preds = []
        all_targets = []

        model.eval()

        with torch.no_grad():
            for data, targets in dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)  # 获取预测值
                all_preds.extend(predicted.tolist())
                all_targets.extend(targets.tolist())

        # 生成混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)

        # 使用 seaborn 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def print_classification_report(model, dataloader):
        all_preds = []
        all_targets = []

        model.eval()

        with torch.no_grad():
            for data, targets in dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.tolist())
                all_targets.extend(targets.tolist())

        # 打印分类报告
        print(classification_report(all_targets, all_preds))



