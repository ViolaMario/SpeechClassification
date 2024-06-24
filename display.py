import mne

n_channels = 64
sfreq = 128  # 采样率
ch_names = [f'EEG {i + 1}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels

# 创建通道信息
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

raw = mne.io.RawArray(data, info)

raw.plot(scalings='auto')

#
n_epochs, n_channels, n_samples = X.shape
event_ids = Y + 1
events = np.column_stack([np.arange(n_epochs), np.zeros(n_epochs, dtype=int), event_ids])
epochs = mne.EpochsArray(X, info, events, tmin=0, event_id=stims)

epochs.plot(picks="eeg", show_scrollbars=False, events=True, scalings='auto')


################
import mne
sfreq = 128  # 采样率
n_epochs, n_channels, n_samples = X.shape
event_ids = Y + 1
ch_names = [f'EEG {i + 1}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
events = np.column_stack([np.arange(n_epochs), np.zeros(n_epochs, dtype=int), event_ids])
epochs = mne.EpochsArray(X, info, events, tmin=0, event_id=stims)

# Display
epochs[0].plot(scalings='auto')    # 查看第一个epoch的时间序列
epochs.average().plot()
epochs.plot_psd(fmax=50)
epochs['1'].average().plot() # class = '1'

# 检查训练集和测试集是否有交叉
# 转换为不可变的形式以便进行比较
train_set = set(map(tuple, X_train.reshape(X_train.shape[0], -1)))
test_set = set(map(tuple, X_test.reshape(X_test.shape[0], -1)))
intersection = train_set.intersection(test_set)
assert len(intersection) == 0, "训练集和测试集不应该有交叉"



# 简单的模型示例
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=100)  # 如果有问题，可以增加 max_iter
model.fit(X_train.reshape(3392, -1), Y_train)  # Flatten the last two dimensions
y_pred = model.predict(X_test.reshape(849, -1))

accuracy = accuracy_score(Y_test, y_pred)
print("测试集准确率:", accuracy)
