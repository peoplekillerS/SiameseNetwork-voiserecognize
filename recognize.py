import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.transforms import MFCC
from sklearn.model_selection import train_test_split
import glob
import torch.nn.functional as F
from torchaudio.transforms import MFCC, TimeMasking, FrequencyMasking, Vol


class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout layer with dropout probability 0.5
            nn.Linear(hidden_size, 256),  # Increase the size of the hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout layer with dropout probability 0.5
            nn.Linear(256, 128),  # Output embedding size
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout layer with dropout probability 0.5
            nn.Linear(128, 64),  # Output embedding size
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout layer with dropout probability 0.5
            nn.Linear(64, 1)  # Output embedding size

        )

    def forward_one(self, x):
        x = self.cnn(x)
        # print("CNN output shape:", x.shape)  # Print CNN output shape
        x = x.view(x.size(0), -1)
        # print("Flattened shape:", x.shape)  # Print flattened shape
        x = self.fc(x)
        # print("FC output shape:", x.shape)  # Print FC output shape
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, weight_diff=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight_diff = weight_diff  # 增加不同样本的惩罚权重

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_same = 0.5 * torch.pow(euclidean_distance, 2)
        loss_diff = 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                    2) * self.weight_diff  # 使用权重增加不同样本的惩罚
        loss = target * loss_same + (1 - target) * loss_diff
        return loss.mean()


def preprocess_audio(file_path, target_length, num_mfcc_coefficients=20, num_mel_filters=40):
    waveform, sample_rate = torchaudio.load(file_path)

    # Crop or pad audio to target_length
    if waveform.size(1) < target_length:
        pad_size = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :target_length]
        # 修改MFCC转换的参数
    mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=num_mfcc_coefficients,  # 增加MFCC系数的数量
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mel_filters},  # 设置滤波器的数量
    )
    mfcc = mfcc_transform(waveform)
    return mfcc


def preprocess_audio_with_offset(file_path, target_length, offset_seconds, num_mfcc_coefficients=20,
                                 num_mel_filters=40):
    waveform, sample_rate = torchaudio.load(file_path, num_frames=-1)

    # Calculate the starting sample point for the offset
    offset_samples = int(sample_rate * offset_seconds)

    # Apply offset and ensure the waveform is not shorter than target_length
    if waveform.size(1) - offset_samples < target_length:
        pad_size = target_length - (waveform.size(1) - offset_samples)
        waveform = torch.nn.functional.pad(waveform, (offset_samples, pad_size))
    else:
        waveform = waveform[:, offset_samples:offset_samples + target_length]

    mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=num_mfcc_coefficients,  # 增加MFCC系数的数量
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mel_filters},  # 设置滤波器的数量
    )
    mfcc = mfcc_transform(waveform)
    return mfcc


# Load and preprocess dataset
data_path = "./zwsjj"
file_list = glob.glob(os.path.join(data_path, "*.wav"))

# Load and preprocess dataset
target_length = 200000  # Example: 1 second of audio at 16kHz sample rate
dataset = []
dataset1 = []
# Create positive and negative pairs for training
positive_pairs = []  # Pairs from the same person
negative_pairs = []  # Pairs from different people
sample_rate = 200000
frame_length = int(0.025 * sample_rate)  # 25ms
frame_shift = int(0.010 * sample_rate)  # 10ms
num_mfcc_coefficients = 13
num_frames = int((sample_rate - frame_length) / frame_shift) + 1

# dataset initial 不取后面的秒数 保证完全mask
count = 1
for file_path in file_list:
    label = count
    mfcc = preprocess_audio(file_path, target_length)
    if mfcc.shape[0] == 2:
        # 对通道进行平均，得到新的张量
        new_tensor = torch.mean(mfcc, dim=0, keepdim=True)
    else:
        new_tensor = mfcc
    dataset.append((new_tensor, label))
    count = count + 1

    # print("MFCC shape for file", file_path, ":", new_tensor.shape)  # Print MFCC shape for each file
print('finish')

# dataset1 initial
count = 1
for file_path in file_list:
    label = count
    mfcc = preprocess_audio_with_offset(file_path, target_length, 20)
    if mfcc.shape[0] == 2:
        # 对通道进行平均，得到新的张量
        new_tensor = torch.mean(mfcc, dim=0, keepdim=True)
    else:
        new_tensor = mfcc
    dataset1.append((new_tensor, label))
    count = count + 1
print('finish1')

# positivie pair initial
for i in range(0, 700):
    # if i + 1 < len(dataset):
    positive_pairs.append((dataset[i][0], dataset1[i][0], 1))

for i in range(700, len(dataset) - 1, 2):
    negative_pairs.append((dataset[i][0], dataset[i + 1][0], 0))

#
# # dataset3 initial
# count = 1
# for file_path in file_list1:
#     label = count
#     mfcc = preprocess_audio_with_offset(file_path, target_length, 80)
#     if mfcc.shape[0] == 2:
#         # 对通道进行平均，得到新的张量
#         new_tensor = torch.mean(mfcc, dim=0, keepdim=True)
#     else:
#         new_tensor = mfcc
#     dataset3.append((new_tensor, label))
#     count = count + 1
# print('finish2')
# count = 1
# # dataset1 initial
# for file_path in file_list:
#     label = count
#     mfcc = preprocess_audio(file_path, target_length)
#     if mfcc.shape[0] == 2:
#         # 对通道进行平均，得到新的张量
#         new_tensor = torch.mean(mfcc, dim=0, keepdim=True)
#     else:
#         new_tensor = mfcc
#     dataset1.append((new_tensor, label))
#     count = count + 1
#     # print("MFCC shape for file", file_path, ":", mfcc.shape)  # Print MFCC shape for each file
# print('finish3')
#
#
#
#
# test_set=[]
# # positive pair initial
# for i in range(0, 500):
#     # if i + 1 < len(dataset):
#     positive_pairs.append((dataset[i][0], dataset1[i][0], 1))
#     positive_pairs.append((dataset2[i][0], dataset3[i][0], 1))
#     positive_pairs.append((dataset[i][0], dataset2[i][0], 1))
#     positive_pairs.append((dataset1[i][0], dataset3[i][0], 1))
#     positive_pairs.append((dataset1[i][0], dataset2[i][0], 1))
#     positive_pairs.append((dataset[i][0], dataset3[i][0], 1))
#
#
#
# for i in range(500, len(dataset)):
#     # if i + 1 < len(dataset):
#     test_set.append((dataset[i][0], dataset1[i][0], 1))
#     test_set.append((dataset[i][0], dataset3[i][0], 1))
#     test_set.append((dataset2[i][0], dataset3[i][0], 1))
#     test_set.append((dataset[i][0], dataset2[i][0], 1))
#     test_set.append((dataset1[i][0], dataset3[i][0], 1))
#
#
# # negative pair initial
# count_2 = 0
# flag = False
# for i in range(0, 400, 3):
#     for j in range(i + 1, 400, 7):
#         negative_pairs.append((dataset[i][0], dataset1[j][0], 0))
#         count_2 = count_2 + 1
#         if count_2 >= 800:
#             flag = True
#             break
#     if flag:
#         break
# count_2 = 0
# flag = False
# for i in range(0, 400, 3):
#     for j in range(i + 1, 400, 7):
#         negative_pairs.append((dataset2[i][0], dataset3[j][0], 0))
#         count_2 = count_2 + 1
#         if count_2 >= 800:
#             flag = True
#             break
#     if flag:
#         break
# count_2 = 0
# flag = False
# for i in range(0, 400, 3):
#     for j in range(i + 1, 400, 7):
#         negative_pairs.append((dataset1[i][0], dataset3[j][0], 0))
#         count_2 = count_2 + 1
#         if count_2 >= 800:
#             flag = True
#             break
#     if flag:
#         break
# count_2 = 0
# flag = False
# for i in range(0, 400, 3):
#     for j in range(i + 1, 400, 7):
#         negative_pairs.append((dataset1[i][0], dataset2[j][0], 0))
#         count_2 = count_2 + 1
#         if count_2 >= 800:
#             flag = True
#             break
#     if flag:
#         break
# count_2 = 0
# for i in range(400, len(dataset), 2):
#     for j in range(i + 1, len(dataset), 2):
#         test_set.append((dataset1[i][0], dataset1[j][0], 0))
#         count_2 = count_2 + 1
#         if count_2 >= 1200:
#             flag = True
#             break
#     if flag:
#         break
#


# Combine positive and negative pairs
all_pairs = positive_pairs + negative_pairs

# # Split dataset
train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2)

print('len(positive_pairs)', len(positive_pairs))
print('len(negative_pairs)', len(negative_pairs))
print('len(train_pairs)', len(train_pairs))
print('len(test_pairs)', len(test_pairs))

batch_size = 32
# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_pairs, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_pairs, batch_size=batch_size, shuffle=True)

# Instantiate the Siamese Network model
input_size = 19968
hidden_size = 256
model = SiameseNetwork(input_size=input_size, hidden_size=hidden_size)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = ContrastiveLoss(margin=1, weight_diff=1)

optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    for input1, input2, target in train_dataloader:
        optimizer.zero_grad()
        # Move data to GPU
        input1, input2, target = input1.to(device), input2.to(device), target.to(device)
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, target.float())
        # print('train-loss:', loss)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}] Training complete")
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        count_3 = 0
        for input1, input2, target in test_dataloader:
            # Move data to GPU
            input1, input2, target = input1.to(device), input2.to(device), target.to(device)
            output1, output2 = model(input1, input2)
            predictions = (output1 - output2).abs() < 0.05
            predicted_labels = predictions.int().squeeze()
            correct += (predicted_labels == target).sum().item()
            total += target.size(0)
            print('correct', correct)
            print('total', total)
            print('target', target)
            print('pre-label', predicted_labels)
        test_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}")
torch.save(model.state_dict(), "new-1.pth")
print("Trained model saved")
