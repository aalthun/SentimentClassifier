import pandas as pd
import json
from collections import Counter

def get_stats(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Tokenizer:
    def __init__(self, merges):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.pad_token = "[PAD]"
        self.pad_token_id = 256
        self.vocab[self.pad_token_id] = self.pad_token.encode("utf8")
        current_token_id = self.pad_token_id + 1
        self.merges = {}
        for key, _ in merges.items():
            if isinstance(key, str):  
                pair = tuple(key.split())  
            else:
                pair = key  
            self.merges[pair] = current_token_id
            try:
                self.vocab[current_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            except KeyError as e:
                print(f"Key {e} could not be found. Vocab keys must be integers.")
                raise
            current_token_id += 1

    def decode(self, ids):
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf8", errors="replace")

    def encode(self, text):
        if not text:
            return []
        tokens = list(text.encode("utf8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            tokens = merge(tokens, pair, new_id)
        return tokens

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    t_string = ' '.join(df['Text'])
    tokens = list(t_string.encode('utf8'))

    ids = tokens
    merges_generated = {}
    idx = 257  
    while True:
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        if stats[pair] < 2:
            break
        print(f"Merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges_generated[pair] = idx
        idx += 1

    serializable_merges = {str(k): v for k, v in merges_generated.items()}
    with open("merges.json", "w", encoding="utf8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=4)
    print("merges saved to tokens.json.")