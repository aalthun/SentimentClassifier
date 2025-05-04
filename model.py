import torch
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tokenizer import Tokenizer
import json
import ast

n_embd = None
n_head = None
vocab_size = None
seq_len = None
n_block = None
num_classes = None
iterations = None
dropout = None
sub_space_size = n_embd//n_head

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

data = pd.read_csv('data.csv')
with open("tokens.json", "r", encoding="utf-8") as f:
    merges = json.load(f)

merges = {ast.literal_eval(k): int(v) for k, v in merges.items()}

class ReviewData(Dataset):
    def __init__(self, data_source, seq_len, split, tokenizer, isTrain=True):
        super().__init__()
        self.data = data_source
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        split_index = int(len(self.data) * split)
        self.train_data, self.valid_data = train_test_split(self.data, test_size=split_index)
    
    def _process_text(self, text):
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.seq_len:
            token_ids = token_ids[:self.seq_len]
        else:
            pad_token = self.tokenizer.pad_token_id  
            token_ids = token_ids + [pad_token] * (self.seq_len - len(token_ids))
        return token_ids
    
    def __getitem__(self, index):
        data = self.train_data if self.isTrain else self.valid_data
        text = data['Review'].iloc[index]
        x = self._process_text(text) 
        sentiments = [
            float(data['Sentiment_Negative'].iloc[index]), 
            float(data['Sentiment_Notr'].iloc[index]), 
            float(data['Sentiment_Positive'].iloc[index])
        ]
        label = sentiments.index(max(sentiments))
        return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        data = self.train_data if self.isTrain else self.valid_data
        return len(data)

class Head(nn.Module):
    def __init__(self, sub_space_size):
        super().__init__()       
        self.Wq = nn.Linear(n_embd, sub_space_size, bias=False)
        self.Wk = nn.Linear(n_embd, sub_space_size, bias=False)
        self.Wv = nn.Linear(n_embd, sub_space_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,mask=None):
        B,T,C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x) 
        scores = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)   
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        a = F.softmax(scores, dim=-1)
        a = self.dropout(a)
        return a @ v     
    
class MultiHeadAttention(nn.Module):
    def __init__(self, sub_space_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(sub_space_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        f_head = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.proj(f_head)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout))
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sub_space_size = n_embd/n_head
        self.attention = MultiHeadAttention(sub_space_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask = None):
        x = self.attention(x, mask)
        x = self.ln1(x) + x
        x = self.ff(x)
        x = self.ln2(x) + x
        return x
    
class SentimentModel(nn.Module):
    def __init__(self, n_block, tokenizer):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_encoding_table =nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embd)
        self.probs = nn.Linear(n_embd, num_classes)
    
    def forward(self, x):
        mask = x != self.pad_token_id 
        a = self.token_embedding_table(x)
        b = self.position_encoding_table(x)
        x = a + b
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        probs = self.probs(x)
        return probs

def calculate_loss(train_loader, valid_loader, model):
    with torch.no_grad():
        model.eval()
        out = {}
        criterion = nn.CrossEntropyLoss()
        for split, loader in zip(['train', 'valid'], [train_loader, valid_loader]):
            losses = []
            for x, y in loader:
                x, y = to_device(x, device), to_device(y, device)
                preds = model(x) 
                preds = preds.mean(dim=1)  
                loss = criterion(preds, y)        
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out
    
tokenizer = Tokenizer(merges)
train_data = ReviewData(data_source=data, tokenizer=tokenizer, seq_len=10, split=0.8)
valid_data = ReviewData(data_source=data, tokenizer=tokenizer, seq_len=10, split=0.8, isTrain=False)

model = SentimentModel(n_block, tokenizer=tokenizer).to(device)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

for i in range(iterations):
    optimizer.zero_grad(set_to_none=True)
    x, y = next(iter(train_loader))
    x, y = to_device(x, device), to_device(y, device) 
    preds = model(x)
    B, T, C = preds.shape
    preds = preds.mean(dim=1)  
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    
    losses = calculate_loss(train_loader, valid_loader, model)
    print(f"Iter {i}: Train Loss = {losses['train']:.4f}, Valid Loss = {losses['valid']:.4f}")