import torch
import torch.nn as nn
from torch.nn import functional as F



# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128 # 32 =  head_size = n_embd / n_head
n_head = 8
n_layer = 20
dropout = 0.2
# ------------

#torch.manual_seed(1337)


!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    #print(text)
    
    
    
print(len(text))



# here are all the unique characters that occur in this text
print(set(text))
print(list(set(text)))
print(sorted(list(set(text))))
print(len(sorted(list(set(text)))))
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(stoi['\n'])
print(encode(['A','B','C']))
print(decode(encode(['A','B','C'])))




# to do use tiktoken to process the text
!pip install tiktoken


import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")

print(enc.encode("Hello World!!!"))
print(enc.n_vocab)




# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(train_data[0:100])
print(text[0:100])
print(sorted(list(set(text))))


#block_size = 8
print(train_data[0:block_size+1])
print(text[0:block_size+1])


x = train_data[0:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
  context = x[0:t+1]
  target = y[t]
  #print(context,"->",target)
  
  
  
  
  
  # data loading
#torch.manual_seed(1337)
#batch_size = 4
#block_size = 8

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')
print("input:")
print(xb.shape)
print(xb)
print("target:")
print(yb.shape)
print(yb)

for b in range(batch_size):
  for t in range(block_size):
    context = xb[ b, 0:t+1 ]
    target = yb[ b, t ]
    #print(context,"->",target)
    
    
    
    
    
    class GateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GateLayer, self).__init__()
        
        self.linear = nn.Linear(input_dim, n_embd)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoder_output):
        # Compute gating vector
        gating_vector = self.linear(encoder_output)
        gating_vector = self.sigmoid(gating_vector)
        # Apply gating vector to encoder output
        gated_encoder_output = encoder_output * gating_vector
        
        return gated_encoder_output



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key  = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.gate = GateLayer( n_embd, head_size )


    def forward(self, x):
        B,T,C = x.shape
        #print("====x=====")
        #print(x.shape)
        x = self.gate(x)

        k = self.key(x)   # (B,T,C)
        #print(k.shape)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class GPT2(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.ffwd = FeedFoward(n_embd)
        #self.sa_head = MultiHeadAttention(4,n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):#target (B,T)
        #B,T =idx.shape
        #tok_embed = self.token_embedding_table(idx) #(B,T,C) (batch,Time,Channel)
        #logits = self.lm_head(tok_embed) #(B,T,vocab_size)

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        #tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C) + pos_emb across the batch
        #x = self.blocks(x) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        
        #x = self.sa_head(x) # (B,T,vocab_size)
        #logits = self.lm_head(x) # (B,T,vocab_size)

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C)
        #x = self.blocks(x) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        #x = self.sa_head(x) # (B,T,vocab_size)
        #x = self.ffwd(x)
        #logits = self.lm_head(x) # (B,T,vocab_size)

        x = tok_emb + pos_emb # (B,T,C) + pos_emb across the batch
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        #x = self.sa_head(x) # (B,T,vocab_size)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets == None:
          loss = None
        else:
          B, T, C = logits.shape # logit(p) = ln( p / ( 1 - p ) )
          logits = logits.view(B*T,C)
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits,targets) # H( P, Q ) = -0.9 * log( 0.8 ) - 0.1 * log( 0.2 ) = 0.311, the lower the better matching
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


'''
model.load_state_dict(torch.load("./python_model"))
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
'''


model = GPT2()
m = model.to(device)
logits, loss = m(xb,yb)
print(logits.shape,loss) # -ln(1/65) = 4.174387269896




idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))


# create a PyTorch optimizer
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


for iter in range(5000):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())



torch.save(m.state_dict(), "./GPT2_model")


print(decode(m.generate(torch.zeros((1,1), dtype=torch.long,device=device), max_new_tokens=1000)[0].tolist()))


'''
#torch.manual_seed(1337)
B,T,C = 4,8,32
x = torch.randn(B,T,C)

head_size = 16
key  = nn.Linear(C,head_size,bias=False)
query = nn.Linear(C,head_size,bias=False)
value = nn.Linear(C,head_size,bias=False)

k = key(x) #(B,T,16)
q = query(x) #(B,T,16)
v = value(x)

wei = q @ k.transpose(-2,-1) # (B,T,16) @ (B,16,T) => (B,T,T)
tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0,float('-inf'))
wei = F.softmax(wei,dim=-1)

output = wei @ v
output.shape
wei[0]



torch.softmax(torch.tensor([0.1,-0.2,0.3,-0.2,0.5]),dim=-1)

torch.softmax(torch.tensor([0.1,-0.2,0.3,-0.2,0.5])*8,dim=-1)

'''











































    
    
    
