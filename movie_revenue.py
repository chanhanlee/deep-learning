import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df1_path, df2_path):
    df8 = pd.read_csv(df1_path)
    df = pd.read_csv(df2_path)

    # 計算頻率
    freq = df8['production_companies'].value_counts(normalize=True)
    df8['production_companies_freq'] = df8['production_companies'].map(freq)

    freq1 = df8['genres'].value_counts(normalize=True)
    df8['genres_freq'] = df8['genres'].map(freq1)

    # 合併數據
    merged_df = pd.merge(df, df8, on='movie_id', how='inner')

    # 選擇年份大於2000的數據
    df2 = merged_df[merged_df['year_x'] > 2000]

    # 選擇所需特徵
    X = df2[['sequal_x', 'reunion_holiday', 'non_reunion_holiday', 'budget_transfer_x', 'production_companies_freq', 'genres_freq']]
    y = df2['revenue_transfer_x']

    # 將類別特徵轉換為整數索引
    X['production_companies'] = X['production_companies_freq'].astype('category').cat.codes
    X['genres'] = X['genres_freq'].astype('category').cat.codes

    # 移除原始類別列
    X.drop(columns=['production_companies_freq', 'genres_freq'], inplace=True)

    # 標準化特徵
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 數據轉換為 PyTorch 張量
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y.values / 1.e8, dtype=torch.float).unsqueeze(1)

    processed_data = list(zip(X_tensor, y_tensor))

    return processed_data

class MovieDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MovieModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(MovieModel, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size3),
            nn.Linear(hidden_size3, 1)
        )

    def forward(self, x):
        return self.fc(x)

torch.manual_seed(42)

# 讀取並處理數據
# 讀取並處理數據
processed_data = preprocess_data('fin_imdb_merge_df_clean.csv', 'df_condensed.csv')

# 將數據集分為訓練集和測試集
train_ratio = 0.8
train_size = int(len(processed_data) * train_ratio)
test_size = len(processed_data) - train_size
train_dataset, test_dataset = random_split(processed_data, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_size = 6  # 根據您提供的特徵數量
hidden_size1 = 256
hidden_size2 = 128
hidden_size3 = 64
model = MovieModel(input_size, hidden_size1, hidden_size2, hidden_size3)
loss = nn.MSELoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練和測試模型
num_epochs = 100
train_loss_list = []
test_loss_list = []
for epoch in range(num_epochs):
    model.train()
    
    train_loss = 0
    for inputs, targets in train_loader:
        # 歸零梯度
        optimizer.zero_grad()
        
        # 向前傳播
        outputs = model(inputs)
        
        # 計算損失
        loss = criterion(outputs, targets)
        
        # 向後傳播
        loss.backward()
        
        # 更新梯度
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)

    model.eval()
    test_loss = sum(criterion(model(inputs), targets).item() for inputs, targets in test_loader) / len(test_loader)
    test_loss_list.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

plt.plot(range(1, num_epochs+1), train_loss_list, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 計算 R 平方值
y_true_train = []
y_pred_train = []
for inputs, targets in train_dataset:
    y_true_train.append(targets.item())
    y_pred_train.append(model(inputs.unsqueeze(0)).item())
y_true_train = np.array(y_true_train)
y_pred_train = np.array(y_pred_train)
r2_train = r2_score(y_true_train, y_pred_train)

y_true_test = []
y_pred_test = []
for inputs, targets in test_dataset:
    y_true_test.append(targets.item())
    y_pred_test.append(model(inputs.unsqueeze(0)).item())
y_true_test = np.array(y_true_test)
y_pred_test = np.array(y_pred_test)
r2_test = r2_score(y_true_test, y_pred_test)

print(f"R2 Score (Train): {r2_train:.4f}")
print(f"R2 Score (Test): {r2_test:.4f}")

