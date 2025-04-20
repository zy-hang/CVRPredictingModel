import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle
import json
from tqdm import tqdm
from CVR_Model import PLE

DATA_PATH = "./DATA/train_user"
CSV_FILE = "tianchi_mobile_recommend_train_user.csv"
print("1) 读取 CSV...")
df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILE))

print("2) 数据预处理...")
# 时间格式转换
df['time'] = pd.to_datetime(df['time'])
df['weekday'] = df['time'].dt.weekday  # 0-6
df['is_weekend'] = (df['weekday'] >= 5).astype(int)  # 是否周末
df['hour'] = df['time'].dt.hour  # 小时: 0-23

# 按用户ID和时间排序
df = df.sort_values(['user_id', 'time'])

# 标签编码
encoders = {}
for col in ['user_id', 'item_id', 'item_category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 创建时间特征
# 计算参考时间点
reference_time = df['time'].max()
df['time_diff'] = (reference_time - df['time']).dt.total_seconds() / 86400  # 转换为天数
scaler = MinMaxScaler()
df['time_diff_norm'] = scaler.fit_transform(df['time_diff'].values.reshape(-1, 1))
df['timestamp_norm'] = (df['time'].astype(int) / 10 ** 9 - df['time'].astype(int).min() / 10 ** 9) / (
        df['time'].astype(int).max() / 10 ** 9 - df['time'].astype(int).min() / 10 ** 9)

print("3) 创建用户行为序列特征...")


def create_user_behavior_counts(df, n_days=7):
    """为每个用户创建最近1-7天的行为计数特征"""
    behavior_counts = []
    for user_id in tqdm(df['user_id'].unique()):
        user_df = df[df['user_id'] == user_id].sort_values('time')
        # 获取用户的所有记录
        user_records = []

        for _, row in user_df.iterrows():
            current_time = row['time']
            # 计算每个行为类型在不同天数窗口内的计数
            counts = {}
            for day in range(1, n_days + 1):
                time_threshold = current_time - pd.Timedelta(days=day)
                past_df = user_df[(user_df['time'] >= time_threshold) & (user_df['time'] < current_time)]

                for behavior in range(1, 5):  # 1,2,3,4
                    count = len(past_df[past_df['behavior_type'] == behavior])
                    counts[f'behavior_{behavior}_days_{day}'] = count

            record = {
                'user_id': user_id,
                'time': current_time,
                'item_id': row['item_id'],
                'item_category': row['item_category'],
                'behavior_type': row['behavior_type'],
                **counts
            }
            user_records.append(record)

        behavior_counts.extend(user_records)

    behavior_df = pd.DataFrame(behavior_counts)
    return behavior_df


# 创建行为计数特征
behavior_counts_df = create_user_behavior_counts(df)
df = pd.merge(df, behavior_counts_df, on=['user_id', 'time', 'item_id', 'item_category', 'behavior_type'], how='left')

# 填充缺失值
behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
df[behavior_cols] = df[behavior_cols].fillna(0)

print("4) 创建用户行为序列特征...")


class SequenceGenerator:
    def __init__(self, df, max_seq_length=20):
        self.df = df
        self.max_seq_length = max_seq_length
        self.user_sequences = {}
        self._generate_sequences()

    def _generate_sequences(self):
        # 按用户ID分组
        for user_id, group in tqdm(self.df.groupby('user_id')):
            # 按时间排序
            sorted_group = group.sort_values('time')

            # 为每条记录创建前面的行为序列
            sequences = []
            for i in range(len(sorted_group)):
                # 获取当前记录之前的所有行为
                prev_rows = sorted_group.iloc[:i].tail(self.max_seq_length)

                # 提取序列特征
                seq_items = prev_rows['item_id'].tolist()
                seq_categories = prev_rows['item_category'].tolist()
                seq_timestamps = prev_rows['timestamp_norm'].tolist()
                seq_behaviors = prev_rows['behavior_type'].tolist()

                # 填充序列到固定长度
                pad_length = self.max_seq_length - len(seq_items)
                if pad_length > 0:
                    seq_items = [0] * pad_length + seq_items
                    seq_categories = [0] * pad_length + seq_categories
                    seq_timestamps = [0] * pad_length + seq_timestamps
                    seq_behaviors = [0] * pad_length + seq_behaviors

                sequences.append({
                    'seq_items': seq_items,
                    'seq_categories': seq_categories,
                    'seq_timestamps': seq_timestamps,
                    'seq_behaviors': seq_behaviors,
                })

            self.user_sequences[user_id] = sequences

    def get_sequence(self, user_id, index):
        """获取特定用户特定索引的序列"""
        return self.user_sequences.get(user_id, [])[index] if user_id in self.user_sequences and index < len(
            self.user_sequences[user_id]) else None


# 创建用户行为序列
seq_generator = SequenceGenerator(df)

print("5) 创建模型所需的嵌入表...")

# 创建用户和商品的嵌入表
embedding_dims = {
    'user_id': 128,
    'item_id': 128,
    'item_category': 64,
    'behavior_type': 8,
    'weekday': 3,
    'hour': 5
}

# 创建嵌入表
embedding_tables = {}
for feature, dim in embedding_dims.items():
    num_unique = df[feature].nunique() + 1  # +1 是为了留出0作为padding
    embedding_tables[feature] = nn.Embedding(num_unique, dim, padding_idx=0)

# 保存长期兴趣模拟
print("6) 模拟长期兴趣...")

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0


def generate_long_term_interest(df, dim=128):
    """从Redis获取用户长期兴趣向量"""
    user_interests = {}

    try:
        # 连接Redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

        # 获取所有用户ID
        user_ids = df['user_id'].unique()

        for user_id in user_ids:
            # 构建用户在Redis中的键名
            key = f"user:{user_id}"

            # 尝试从Redis获取用户向量
            user_vector = r.get(key)

            if user_vector:
                # 解析JSON字符串为Python列表
                vector = json.loads(user_vector)
                user_interests[user_id] = np.array(vector, dtype=np.float32)
            else:
                # 如果Redis中没有该用户数据，则生成随机向量作为后备
                user_interests[user_id] = np.random.randn(dim).astype(np.float32)

    except Exception as e:
        print(f"从Redis获取用户兴趣向量失败: {e}")
        # 发生异常时回退到随机生成
        for user_id in df['user_id'].unique():
            user_interests[user_id] = np.random.randn(dim).astype(np.float32)

    return user_interests


long_term_interests = generate_long_term_interest(df)

print("7) 构建训练数据集...")


# SE (Squeeze and Excitation) 操作
class SEBlock(nn.Module):
    def __init__(self, field_dims, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.field_dims = field_dims
        self.fc1 = nn.Linear(len(field_dims), len(field_dims) // reduction_ratio)
        self.fc2 = nn.Linear(len(field_dims) // reduction_ratio, len(field_dims))
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        # 每个field的平均值
        field_avgs = [emb.mean(dim=1) for emb in embeddings]
        field_avgs = torch.stack(field_avgs, dim=1)  # [batch_size, num_fields]

        # SE操作
        squeeze = self.fc1(field_avgs)
        squeeze = self.activation(squeeze)
        excitation = self.fc2(squeeze)
        excitation = self.sigmoid(excitation)  # [batch_size, num_fields]

        # 对每个field应用权重
        weighted_embeddings = []
        for i, emb in enumerate(embeddings):
            weight = excitation[:, i].unsqueeze(1)
            weighted_emb = emb * weight.unsqueeze(2)
            weighted_embeddings.append(weighted_emb)

        return weighted_embeddings


class DIN(nn.Module):
    def __init__(self, item_embedding_dim, cat_embedding_dim, behavior_embedding_dim, time_embedding_dim,
                 hidden_size=128):
        super(DIN, self).__init__()
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(item_embedding_dim * 2 + cat_embedding_dim * 2 + time_embedding_dim + behavior_embedding_dim,
                      hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, item_emb, cat_emb, seq_items_emb, seq_cats_emb, seq_times_emb, seq_behaviors_emb, mask):
        # 将当前商品的嵌入扩展到序列长度
        batch_size, seq_len, _ = seq_items_emb.shape
        expanded_item_emb = item_emb.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_cat_emb = cat_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # 拼接特征用于注意力计算
        concat_features = torch.cat([
            expanded_item_emb, seq_items_emb,  # 当前商品和序列商品
            expanded_cat_emb, seq_cats_emb,  # 当前类目和序列类目
            seq_times_emb, seq_behaviors_emb  # 序列时间和行为
        ], dim=2)

        # 计算注意力分数
        attention_score = self.attention(concat_features)  # [batch_size, seq_len, 1]

        # 应用mask
        attention_score = attention_score.squeeze(-1) * mask
        attention_score = attention_score / (attention_score.sum(dim=1, keepdim=True) + 1e-8)

        # 加权求和得到用户的短期兴趣
        weighted_seq_items = (seq_items_emb * attention_score.unsqueeze(-1)).sum(dim=1)
        weighted_seq_cats = (seq_cats_emb * attention_score.unsqueeze(-1)).sum(dim=1)

        # 拼接得到最终的短期兴趣表示
        short_term_interest = torch.cat([weighted_seq_items, weighted_seq_cats], dim=1)

        return short_term_interest


class ECommerceDataset(Dataset):
    def __init__(self, df, seq_generator, long_term_interests, embedding_tables, train=True):
        self.df = df
        self.seq_generator = seq_generator
        self.long_term_interests = long_term_interests
        self.embedding_tables = embedding_tables
        self.train = train

        # 记录合法样本的索引
        self.valid_indices = []
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id]
            for i in range(len(user_df)):
                seq = seq_generator.get_sequence(user_id, i)
                if seq is not None:
                    self.valid_indices.append((user_id, i))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        user_id, index = self.valid_indices[idx]
        user_df = self.df[self.df['user_id'] == user_id].reset_index(drop=True)
        row = user_df.iloc[index]

        # 获取行为序列
        seq = self.seq_generator.get_sequence(user_id, index)

        # 特征提取与嵌入
        features = {
            'user_id': row['user_id'],
            'item_id': row['item_id'],
            'item_category': row['item_category'],
            'weekday': row['weekday'],
            'is_weekend': row['is_weekend'],
            'hour': row['hour'],
            'time_diff_norm': row['time_diff_norm'],
            'timestamp_norm': row['timestamp_norm'],
            'behavior_counts': np.array([row[f'behavior_{b}_days_{d}'] for b in range(1, 5) for d in range(1, 8)]),
            'long_term_interest': self.long_term_interests[user_id],
            'seq_items': np.array(seq['seq_items']),
            'seq_categories': np.array(seq['seq_categories']),
            'seq_timestamps': np.array(seq['seq_timestamps']),
            'seq_behaviors': np.array(seq['seq_behaviors']),
            'behavior_type': row['behavior_type']
        }

        # 构建标签
        # 注意连续行为的处理:
        # 如果之前是购物车/收藏，然后是购买，我们将y1设为1
        # 这可以通过检查当前行为和上一个行为来实现
        behavior_type = row['behavior_type']

        # 对于训练集，构建标签
        if self.train:
            y1 = 1 if behavior_type == 4 else 0  # 是否购买
            y2 = 1 if behavior_type in [2, 3] else 0  # 是否收藏或加购物车

            # 处理连续行为的情况
            if index > 0 and behavior_type == 4:
                prev_behavior = user_df.iloc[index - 1]['behavior_type']
                # 如果前一个行为是收藏或加购物车，当前是购买，则这可能是一个转化路径
                if prev_behavior in [2, 3]:
                    # 可以在这里加入额外的标记或权重，例如y1权重可以更高
                    pass
        else:
            # 对于测试集，没有标签
            y1 = 0
            y2 = 0

        return features, np.array([y1, y2], dtype=np.float32)


def collate_fn(batch):
    features_batch = []
    labels_batch = []

    for features, labels in batch:
        features_batch.append(features)
        labels_batch.append(labels)

    # 转换为张量
    labels_tensor = torch.tensor(np.stack(labels_batch), dtype=torch.float)

    # 组织特征
    user_ids = torch.tensor([f['user_id'] for f in features_batch], dtype=torch.long)
    item_ids = torch.tensor([f['item_id'] for f in features_batch], dtype=torch.long)
    item_categories = torch.tensor([f['item_category'] for f in features_batch], dtype=torch.long)
    weekdays = torch.tensor([f['weekday'] for f in features_batch], dtype=torch.long)
    is_weekends = torch.tensor([f['is_weekend'] for f in features_batch], dtype=torch.float).unsqueeze(1)
    hours = torch.tensor([f['hour'] for f in features_batch], dtype=torch.long)
    time_diff_norms = torch.tensor([f['time_diff_norm'] for f in features_batch], dtype=torch.float).unsqueeze(1)
    timestamp_norms = torch.tensor([f['timestamp_norm'] for f in features_batch], dtype=torch.float).unsqueeze(1)
    behavior_counts = torch.tensor(np.stack([f['behavior_counts'] for f in features_batch]), dtype=torch.float)
    long_term_interests = torch.tensor(np.stack([f['long_term_interest'] for f in features_batch]), dtype=torch.float)

    # 序列特征
    seq_items = torch.tensor(np.stack([f['seq_items'] for f in features_batch]), dtype=torch.long)
    seq_categories = torch.tensor(np.stack([f['seq_categories'] for f in features_batch]), dtype=torch.long)
    seq_timestamps = torch.tensor(np.stack([f['seq_timestamps'] for f in features_batch]), dtype=torch.float)
    seq_behaviors = torch.tensor(np.stack([f['seq_behaviors'] for f in features_batch]), dtype=torch.long)

    # 序列掩码 (用于DIN中的注意力机制)
    mask = (seq_items > 0).float()

    return {
               'user_ids': user_ids,
               'item_ids': item_ids,
               'item_categories': item_categories,
               'weekdays': weekdays,
               'is_weekends': is_weekends,
               'hours': hours,
               'time_diff_norms': time_diff_norms,
               'timestamp_norms': timestamp_norms,
               'behavior_counts': behavior_counts,
               'long_term_interests': long_term_interests,
               'seq_items': seq_items,
               'seq_categories': seq_categories,
               'seq_timestamps': seq_timestamps,
               'seq_behaviors': seq_behaviors,
               'mask': mask
           }, labels_tensor


print("8) 数据集划分...")
# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建数据集
train_dataset = ECommerceDataset(train_df, seq_generator, long_term_interests, embedding_tables)
val_dataset = ECommerceDataset(val_df, seq_generator, long_term_interests, embedding_tables)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

print("9) 准备训练模型...")

# DIN和SE块
din_model = DIN(
    item_embedding_dim=embedding_dims['item_id'],
    cat_embedding_dim=embedding_dims['item_category'],
    behavior_embedding_dim=embedding_dims['behavior_type'],
    time_embedding_dim=1
)

se_block = SEBlock([
    embedding_dims['item_id'],
    embedding_dims['user_id'],
    embedding_dims['item_category'],
    11,  # 时间特征总和
    28,  # 用户行为计数特征
    128,  # 用户长期兴趣
    128 + 64  # DIN输出的短期兴趣 (item_emb + cat_emb)
])

# 初始化CVR预测模型
model = PLE(input_dim=615)

# 优化器和损失函数
optimizer = optim.Adam(list(model.parameters()) + list(din_model.parameters()) + list(se_block.parameters()) +
                       [p for tables in embedding_tables.values() for p in tables.parameters()],
                       lr=0.001)
criterion = nn.BCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将模型移至设备
model.to(device)
din_model.to(device)
se_block.to(device)
for table in embedding_tables.values():
    table.to(device)

print("10) 开始训练...")


def train_epoch(model, din_model, se_block, embedding_tables, train_loader, optimizer, criterion, device):
    model.train()
    din_model.train()
    se_block.train()

    total_loss = 0
    y1_correct = 0
    y2_correct = 0
    total_samples = 0

    for batch_idx, (features, labels) in enumerate(tqdm(train_loader)):
        # 将数据移至设备
        for key in features:
            features[key] = features[key].to(device)
        labels = labels.to(device)

        # 提取嵌入
        user_emb = embedding_tables['user_id'](features['user_ids'])
        item_emb = embedding_tables['item_id'](features['item_ids'])
        cat_emb = embedding_tables['item_category'](features['item_categories'])

        # 时间特征嵌入
        weekday_emb = embedding_tables['weekday'](features['weekdays'])
        hour_emb = embedding_tables['hour'](features['hours'])
        is_weekend = features['is_weekends']
        time_diff = features['time_diff_norms']
        timestamp = features['timestamp_norms']

        # 将时间特征合并
        time_features = torch.cat([weekday_emb, is_weekend, hour_emb, time_diff, timestamp], dim=1)

        # 用户行为计数
        behavior_counts = features['behavior_counts']

        # 长期兴趣
        long_term_interest = features['long_term_interests']

        # 序列特征嵌入
        seq_items_emb = embedding_tables['item_id'](features['seq_items'])
        seq_cats_emb = embedding_tables['item_category'](features['seq_categories'])
        seq_times_emb = features['seq_timestamps'].unsqueeze(-1)
        seq_behaviors_emb = embedding_tables['behavior_type'](features['seq_behaviors'])

        # 使用DIN计算短期兴趣
        short_term_interest = din_model(
            item_emb, cat_emb,
            seq_items_emb, seq_cats_emb,
            seq_times_emb, seq_behaviors_emb,
            features['mask']
        )

        # 收集所有特征场
        field_embeddings = [
            item_emb,
            user_emb,
            cat_emb,
            time_features,
            behavior_counts,
            long_term_interest,
            short_term_interest
        ]

        # 应用SE操作
        weighted_embeddings = se_block(field_embeddings)

        # 拉平并连接所有特征
        flat_embeddings = []
        for emb in weighted_embeddings:
            flat_embeddings.append(emb.flatten(start_dim=1))

        combined_embeddings = torch.cat(flat_embeddings, dim=1)

        # 模型预测
        y1_pred, y2_pred = model(combined_embeddings)

        # 计算损失
        loss_y1 = criterion(y1_pred, labels[:, 0:1])
        loss_y2 = criterion(y2_pred, labels[:, 1:2])
        loss = loss_y1 + loss_y2

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * labels.size(0)
        y1_correct += ((y1_pred > 0.5).float() == labels[:, 0:1]).sum().item()
        y2_correct += ((y2_pred > 0.5).float() == labels[:, 1:2]).sum().item()
        total_samples += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, y1 acc: {y1_correct / total_samples:.4f}, y2 acc: {y2_correct / total_samples:.4f}')

    return total_loss / total_samples, y1_correct / total_samples, y2_correct / total_samples


def validate(model, din_model, se_block, embedding_tables, val_loader, criterion, device):
    model.eval()
    din_model.eval()
    se_block.eval()

    total_loss = 0
    y1_correct = 0
    y2_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in tqdm(val_loader):
            # 将数据移至设备
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            # 提取嵌入
            user_emb = embedding_tables['user_id'](features['user_ids'])
            item_emb = embedding_tables['item_id'](features['item_ids'])
            cat_emb = embedding_tables['item_category'](features['item_categories'])

            # 时间特征嵌入
            weekday_emb = embedding_tables['weekday'](features['weekdays'])
            hour_emb = embedding_tables['hour'](features['hours'])
            is_weekend = features['is_weekends']
            time_diff = features['time_diff_norms']
            timestamp = features['timestamp_norms']

            # 将时间特征合并
            time_features = torch.cat([weekday_emb, is_weekend, hour_emb, time_diff, timestamp], dim=1)

            # 用户行为计数
            behavior_counts = features['behavior_counts']

            # 长期兴趣
            long_term_interest = features['long_term_interests']

            # 序列特征嵌入
            seq_items_emb = embedding_tables['item_id'](features['seq_items'])
            seq_cats_emb = embedding_tables['item_category'](features['seq_categories'])
            seq_times_emb = features['seq_timestamps'].unsqueeze(-1)
            seq_behaviors_emb = embedding_tables['behavior_type'](features['seq_behaviors'])

            # 使用DIN计算短期兴趣
            short_term_interest = din_model(
                item_emb, cat_emb,
                seq_items_emb, seq_cats_emb,
                seq_times_emb, seq_behaviors_emb,
                features['mask']
            )

            # 收集所有特征场
            field_embeddings = [
                item_emb,
                user_emb,
                cat_emb,
                time_features,
                behavior_counts,
                long_term_interest,
                short_term_interest
            ]

            # 应用SE操作
            weighted_embeddings = se_block(field_embeddings)

            # 拉平并连接所有特征
            flat_embeddings = []
            for emb in weighted_embeddings:
                flat_embeddings.append(emb.flatten(start_dim=1))

            combined_embeddings = torch.cat(flat_embeddings, dim=1)

            # 模型预测
            y1_pred, y2_pred = model(combined_embeddings)

            # 计算损失
            loss_y1 = criterion(y1_pred, labels[:, 0:1])
            loss_y2 = criterion(y2_pred, labels[:, 1:2])
            loss = loss_y1 + loss_y2

            # 统计
            total_loss += loss.item() * labels.size(0)
            y1_correct += ((y1_pred > 0.5).float() == labels[:, 0:1]).sum().item()
            y2_correct += ((y2_pred > 0.5).float() == labels[:, 1:2]).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, y1_correct / total_samples, y2_correct / total_samples


# 训练循环
num_epochs = 5
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练
    train_loss, train_y1_acc, train_y2_acc = train_epoch(
        model, din_model, se_block, embedding_tables, train_loader, optimizer, criterion, device
    )
    print(f"Train Loss: {train_loss:.4f}, Y1 Acc: {train_y1_acc:.4f}, Y2 Acc: {train_y2_acc:.4f}")

    # 验证
    val_loss, val_y1_acc, val_y2_acc = validate(
        model, din_model, se_block, embedding_tables, val_loader, criterion, device
    )
    print(f"Val Loss: {val_loss:.4f}, Y1 Acc: {val_y1_acc:.4f}, Y2 Acc: {val_y2_acc:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"Saving best model with val_loss: {val_loss:.4f}")
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'din_model_state_dict': din_model.state_dict(),
            'se_block_state_dict': se_block.state_dict(),
            'embedding_tables': {k: v.state_dict() for k, v in embedding_tables.items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }, 'best_cvr_model.pth')

print("11) 评估模型...")


def predict(model, din_model, se_block, embedding_tables, test_loader, device):
    model.eval()
    din_model.eval()
    se_block.eval()

    all_y1_preds = []
    all_y2_preds = []

    with torch.no_grad():
        for features, _ in tqdm(test_loader):
            # 将数据移至设备
            for key in features:
                features[key] = features[key].to(device)

            # 提取嵌入
            user_emb = embedding_tables['user_id'](features['user_ids'])
            item_emb = embedding_tables['item_id'](features['item_ids'])
            cat_emb = embedding_tables['item_category'](features['item_categories'])

            # 时间特征嵌入
            weekday_emb = embedding_tables['weekday'](features['weekdays'])
            hour_emb = embedding_tables['hour'](features['hours'])
            is_weekend = features['is_weekends']
            time_diff = features['time_diff_norms']
            timestamp = features['timestamp_norms']

            # 将时间特征合并
            time_features = torch.cat([weekday_emb, is_weekend, hour_emb, time_diff, timestamp], dim=1)

            # 用户行为计数
            behavior_counts = features['behavior_counts']

            # 长期兴趣
            long_term_interest = features['long_term_interests']

            # 序列特征嵌入
            seq_items_emb = embedding_tables['item_id'](features['seq_items'])
            seq_cats_emb = embedding_tables['item_category'](features['seq_categories'])
            seq_times_emb = features['seq_timestamps'].unsqueeze(-1)
            seq_behaviors_emb = embedding_tables['behavior_type'](features['seq_behaviors'])

            # 使用DIN计算短期兴趣
            short_term_interest = din_model(
                item_emb, cat_emb,
                seq_items_emb, seq_cats_emb,
                seq_times_emb, seq_behaviors_emb,
                features['mask']
            )

            # 收集所有特征场
            field_embeddings = [
                item_emb,
                user_emb,
                cat_emb,
                time_features,
                behavior_counts,
                long_term_interest,
                short_term_interest
            ]

            # 应用SE操作
            weighted_embeddings = se_block(field_embeddings)

            # 拉平并连接所有特征
            flat_embeddings = []
            for emb in weighted_embeddings:
                flat_embeddings.append(emb.flatten(start_dim=1))

            combined_embeddings = torch.cat(flat_embeddings, dim=1)

            # 模型预测
            y1_pred, y2_pred = model(combined_embeddings)

            all_y1_preds.extend(y1_pred.cpu().numpy())
            all_y2_preds.extend(y2_pred.cpu().numpy())

    return np.array(all_y1_preds), np.array(all_y2_preds)


# 加载最佳模型
print("12) 加载最佳模型...")
checkpoint = torch.load('best_cvr_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
din_model.load_state_dict(checkpoint['din_model_state_dict'])
se_block.load_state_dict(checkpoint['se_block_state_dict'])
for k, v in checkpoint['embedding_tables'].items():
    embedding_tables[k].load_state_dict(v)

# 评估模型
y1_preds, y2_preds = predict(model, din_model, se_block, embedding_tables, val_loader, device)

# 将预测结果保存为提交文件
print("13) 生成提交文件...")
val_df_reset = val_df.reset_index(drop=True)
val_df_reset['y1_pred'] = y1_preds
val_df_reset['y2_pred'] = y2_preds

# 对于实际提交，你需要按照比赛要求的格式输出
# 假设需要对商品子集P预测
subset_P = val_df_reset[val_df_reset['item_id'].isin(val_df_reset['item_id'].unique()[:100])]

# 创建提交文件
submission = subset_P[['user_id', 'item_id', 'y1_pred']].copy()
submission.columns = ['user_id', 'item_id', 'score']  # 修改列名为比赛要求
submission.to_csv('submission.csv', index=False)

print("14) 训练和评估完成！")

# 模型解释
print("15) 模型解释和分析")


# 特征重要性分析 (通过SE Block的权重)
def analyze_field_importance(se_block, field_names):
    # 获取SE Block的参数
    fc1_weight = se_block.fc1.weight.data.cpu().numpy()
    fc2_weight = se_block.fc2.weight.data.cpu().numpy()

    # 计算每个字段的综合重要性
    importance = np.abs(np.matmul(fc2_weight, fc1_weight))
    importance = importance.mean(axis=0)

    # 创建字段名称与重要性的映射
    field_importance = {field: imp for field, imp in zip(field_names, importance)}

    # 按重要性排序
    sorted_fields = sorted(field_importance.items(), key=lambda x: x[1], reverse=True)

    return sorted_fields


field_names = [
    '商品ID',
    '用户ID',
    '商品类目',
    '时间特征',
    '用户行为计数',
    '用户长期兴趣',
    '用户短期兴趣'
]

field_importance = analyze_field_importance(se_block, field_names)
print("特征重要性排名:")
for field, importance in field_importance:
    print(f"{field}: {importance:.4f}")


# 分析连续行为的影响
# 识别连续行为模式
def analyze_behavior_patterns(df):
    patterns = []
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].sort_values('time')

        if len(user_df) < 2:
            continue

        for i in range(1, len(user_df)):
            prev_behavior = user_df.iloc[i - 1]['behavior_type']
            curr_behavior = user_df.iloc[i]['behavior_type']

            if prev_behavior in [2, 3] and curr_behavior == 4:
                pattern = {
                    'user_id': user_id,
                    'prev_behavior': prev_behavior,
                    'curr_behavior': curr_behavior,
                    'time_diff': (user_df.iloc[i]['time'] - user_df.iloc[i - 1]['time']).total_seconds() / 3600  # 小时
                }
                patterns.append(pattern)

    return pd.DataFrame(patterns)


behavior_patterns = analyze_behavior_patterns(df)
print(f"找到 {len(behavior_patterns)} 个连续行为模式")
if len(behavior_patterns) > 0:
    print("行为转化时间间隔统计:")
    print(behavior_patterns['time_diff'].describe())

    # 统计不同前置行为的分布
    prev_behavior_counts = behavior_patterns['prev_behavior'].value_counts()
    print("\n前置行为分布:")
    for behavior, count in prev_behavior_counts.items():
        behavior_name = "收藏" if behavior == 2 else "加购物车"
        print(f"{behavior_name}: {count} ({count / len(behavior_patterns) * 100:.2f}%)")

print("训练和分析完成！")
