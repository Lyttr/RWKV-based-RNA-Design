
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from rwkv.model import RWKV  # 假设你已经安装了 rwkv-pip 或 rwkv-lm

# 1. Tokenizer (简单字符级)
token_to_id = {
    'A': 0, 'U': 1, 'G': 2, 'C': 3,
    '(': 4, ')': 5, '.': 6,
    '[STRUCTURE]': 7, '[BOS]': 8, '[EOS]': 9
}
id_to_token = {v: k for k, v in token_to_id.items()}

def tokenize(structure, sequence):
    input_ids = [token_to_id["[STRUCTURE]"]] + [token_to_id[c] for c in structure] + [token_to_id["[BOS]"]]
    target_ids = [token_to_id[c] for c in sequence] + [token_to_id["[EOS]"]]
    return input_ids, target_ids

# 2. 数据集加载
class RNADesignDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                inp, tgt = tokenize(item['structure'], item['sequence'])
                self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        x = torch.tensor(inp + tgt[:-1])  # 模拟 GPT-style LM 输入
        y = torch.tensor(tgt)             # 只计算 target 部分 loss
        return x, y

# 3. 简单训练循环
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RNADesignDataset("eterna_filtered.jsonl")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: collate(batch, device))

    model = RWKV.from_pretrained("RWKV-169M-Pile")  # 示例：你应下载一个预训练模型或用 RWKVConfig 自建
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        for step, (x, y) in enumerate(dataloader):
            logits = model(x)  # (B, T, V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"[Epoch {epoch}] Step {step} Loss: {loss.item():.4f}")

def collate(batch, device):
    # padding
    max_len = max(len(x[0]) for x in batch)
    x_batch, y_batch = [], []
    for x, y in batch:
        x_pad = x + [0] * (max_len - len(x))
        y_pad = y + [0] * (max_len - len(y))
        x_batch.append(x_pad)
        y_batch.append(y_pad)
    return torch.tensor(x_batch).to(device), torch.tensor(y_batch).to(device)

if __name__ == "__main__":
    train()
