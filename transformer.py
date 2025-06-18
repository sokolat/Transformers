import torch
from torch import nn
import math
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse
from datasets import Dataset as ds
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--key_dim", type=int, default=64)
    parser.add_argument("--val_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_encoder_layers", type=int, default=64)
    parser.add_argument("--tokens_per_batch", type=int, default=25000)
    parser.add_argument("--vocab_size", type=int, default=37000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=1e-09)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_path", type=str, default="./data/wmt14_translate_de-en_train.csv")
    parser.add_argument("--test_path", type=str, default="./data/wmt14_translate_de-en_test.csv")
    parser.add_argument("--val_path", type=str, default="./data/wmt14_translate_de-en_validation.csv")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda')

    return parser.parse_args()

def positional_encoding(num_tokens, model_dim):

    position = torch.arange(num_tokens).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
    pe = torch.zeros(num_tokens, model_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)

class ScaledDotProdAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProdAttention, self).__init__()

    def forward(self, query, key, value, mask=None):

        mat_mul = query @ torch.t(key)
        scaled = mat_mul / torch.sqrt(torch.tensor(query.size(1)))
        if mask is not None:
            scaled[mask[0], mask[1]] = float('-inf')
        softmax = torch.softmax(scaled, dim=1)
        output = softmax @ value
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, key_dim, val_dim):
        super(MultiHeadAttention, self).__init__()

        assert key_dim == val_dim, f"Key dimension ({key_dim}) must equal value dimension ({val_dim}) for attention mechanism"
        assert key_dim == model_dim // num_heads, f"Key dimension ({key_dim}) must equal model_dim/num_heads ({model_dim}/{num_heads}={model_dim // num_heads})"

        self.heads = nn.ModuleList([Head(model_dim, key_dim, val_dim) for _ in range(num_heads)])
        self.linear_o = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)

    def forward(self, query, key, value, mask):

        heads_concat = torch.cat([head(query, key, value, mask) for head in self.heads], 1)
        output = self.linear_o(heads_concat)

        return output

class Head(nn.Module):
    def __init__(self, model_dim, key_dim, val_dim):
        super(Head, self).__init__()

        self.linear_q = nn.Linear(in_features=model_dim, out_features=key_dim, bias=False)
        self.linear_k = nn.Linear(in_features=model_dim, out_features=key_dim, bias=False)
        self.linear_v = nn.Linear(in_features=model_dim, out_features=val_dim, bias=False)

        self.sdp_att = ScaledDotProdAttention()

    def forward(self, query, key, value, mask):

        query_proj = self.linear_q(query)
        key_proj = self.linear_k(key)
        val_proj = self.linear_v(value)

        head = self.sdp_att(query_proj, key_proj, val_proj, mask)

        return head

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, key_dim, val_dim, hidden_dim):
        super(EncoderLayer, self).__init__()

        self.multi_head_att = MultiHeadAttention(num_heads,model_dim, key_dim, val_dim)
        self.ln1 = nn.LayerNorm(model_dim)

        self.fc1 = nn.Linear(in_features=model_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.linear_q = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.linear_k = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.linear_v = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)

        self.shortcut = nn.Sequential()

    def forward(self, x):

        Q, K, V = self.linear_q(x), self.linear_q(x), self.linear_v(x)
        out = self.multi_head_att(Q, K, V)
        out = out + x
        out = self.ln1(out)

        shortcut = self.shortcut(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + shortcut
        out = self.ln2(out)

        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, key_dim, val_dim, hidden_dim):
        super(DecoderLayer, self).__init__()

        self.multi_head_att1 = MultiHeadAttention(num_heads, model_dim, key_dim, val_dim)
        self.ln1 = nn.LayerNorm(model_dim)
 
        self.multi_head_att2 = MultiHeadAttention(num_heads, model_dim, key_dim, val_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.fc1 = nn.Linear(in_features=model_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=model_dim)
        self.ln3 = nn.LayerNorm(model_dim)

        self.decoder_linear_q1 = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.decoder_linear_q2 = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.decoder_linear_k = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.decoder_linear_v = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)      
        self.encoder_linear_k = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.encoder_linear_v = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)

        self.shortcut = nn.Sequential()
    
    def forward(self, x, encoder_out):

        Q, K, V = self.decoder_linear_q1(x), self.decoder_linear_k(x), self.decoder_linear_v(x)
        mask = torch.triu_indices(row=Q.size(0), col=Q.size(1), offset=1)
        out = self.multi_head_att1(Q, K, V, mask)
        out = out + x
        out = self.ln1(out)

        shortcut = self.shortcut(out)

        Q, K, V = self.decoder_linear_q2(out), self.encoder_linear_k(encoder_out), self.encoder_linear_v(encoder_out)
        out = self.multi_head_att2(Q, K, V)
        out = out + shortcut
        out = self.ln2(out)

        shortcut = self.shortcut(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + shortcut
        out = self.ln3(out)

        return out

class Transformer(nn.Module):
    def __init__(self, num_heads, model_dim, key_dim, val_dim, hidden_dim, num_encoder_layers, vocab_size):
        super(Transformer, self).__init__()

        self.model_dim = model_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)

        self.encoder = nn.ModuleList([EncoderLayer(model_dim, num_heads, key_dim, val_dim, hidden_dim) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(model_dim, num_heads, key_dim, val_dim, hidden_dim) for _ in range(num_encoder_layers)])

    def forward(self, src, target):

        src_embedding = self.embedding(src)
        target_embedding = self.embedding(target)

        src_embedding = src_embedding + positional_encoding(src.size(1), self.model_dim)
        target_embedding = target_embedding + positional_encoding(target.size(1), self.model_dim)

        encoder_out = src_embedding

        for encoder_layer in self.encoder:
            encoder_out = encoder_layer(encoder_out)
        
        decoder_out = target_embedding

        for decoder_layer in self.decoder:
            decoder_out = decoder_layer(decoder_out, encoder_out)
        
        breakpoint()
        
        out = self.linear(decoder_out)
        
        return out

def collate_fn(data):

    src_batch, target_batch = zip(*data)

    src_batch = [torch.tensor(x, dtype=torch.long) for x in src_batch]
    target_batch = [torch.tensor(x, dtype=torch.long) for x in target_batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_batch, batch_first=True, padding_value=0)

    return src_padded, target_padded

class TokenizedDataset(Dataset):

    def __init__(self, data_path, vocab_size, chunk, tokenizer_file=None):

        self.dataset = ds.from_pandas(pd.read_csv(data_path, lineterminator='\n'))
        self.data_path = data_path
        self.chunk = chunk
        self.vocab_size = vocab_size
        self.tokenizer_file = tokenizer_file
        self.tokenizer = self._get_tokenizer()
    
    def _get_tokenizer(self):

        def batch_iterator():
            for lang in ['en', 'de']:
                for i in range(0, len(self.dataset), self.chunk):
                    yield self.dataset[i:i+self.chunk][lang]

        if self.tokenizer_file:
            tokenizer = Tokenizer.from_file(self.tokenizer_file)
        else:
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                show_progress=True,
                special_tokens=['<pad>', '<unk>', '<s>', '</s>']
                )
            
            tokenizer.train_from_iterator(
                batch_iterator(),
                trainer=trainer
            )

            tokenizer.save("bpe_tokenizer.json")
        
        return tokenizer
    
    def __getitem__(self, index):
        
        src_ids = self.tokenizer.encode(str(self.dataset[index]['de'])).ids
        target_ids = self.tokenizer.encode(str(self.dataset[index]['en'])).ids

        return src_ids, target_ids
    
    def __len__(self):

        return len(self.dataset)

class TokenBatchSampler(Sampler):
    def __init__(self, dataset, tokens_per_batch, shuffle=True):
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.dataset = dataset

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        batch = []
        src_token_counter, target_token_counter = 0, 0

        for idx in self.indices:
            src_len, target_len = len(self.dataset[idx][0]), len(self.dataset[idx][1])
            src_token_counter += src_len
            target_token_counter += target_len
            if src_token_counter >= self.tokens_per_batch or target_token_counter >= self.tokens_per_batch:
                if batch:
                    yield batch
                    batch = []
                    src_token_counter, target_token_counter = 0, 0
            batch.append(idx)
        
        if batch:
            yield batch

def train(model, args):

    if args.device == 'cuda':
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = TokenizedDataset(
        data_path=args.train_path,
        vocab_size=args.vocab_size,
        chunk=args.chunk,
        tokenizer_file=args.tokenizer_path
    )

    sampler = TokenBatchSampler(
        dataset=train_dataset,
        tokens_per_batch=args.tokens_per_batch,
        shuffle=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn
    )

    for src, target in train_loader:

        src, target = src.to(device), target.to(device)
        
        output = model(src, target)

def main():
    
    args = get_args()

    model = Transformer(
        args.num_heads,
        args.model_dim,
        args.key_dim,
        args.key_dim,
        args.hidden_dim,
        args.num_encoder_layers,
        args.vocab_size
    )

    train(model, args)

if __name__ == "__main__":
    main()

