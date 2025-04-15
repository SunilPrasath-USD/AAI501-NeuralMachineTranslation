# %%
import torch
import torch.nn as nn
import torch.nn as nn
import numpy  as np
#from Encoder import Encoder
#from Decoder import Decoder
import copy
from torch.nn.functional import log_softmax,pad
import math
from torch.optim.lr_scheduler import  LambdaLR
import spacy
import os
from os.path import exists
import gzip
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
from torch.utils.data import DataLoader
import GPUtil

# %%
RUN_EXAMPLES = True

# %%
torch.cuda.is_available()

# %%
def is_interactive_notebook():
    return __name__ =="__main__"

# %%
def show_example(fn,args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)



# %%
def padding_mask(input):
    # Create mask which marks the zero padding values in the input by a 1
    mask = torch.eq(input, 0)
    mask = mask.float()
    return mask

# %%
input = np.array([1,2,3,4,0,0,0])

# %%
padding_mask(torch.tensor(input))

# %%
msk  =  1 - torch.tril(torch.ones(5, 5))

# %%
msk

# %%
msk.shape

# %%
msk = msk.unsqueeze(1)

# %%
msk

# %%
msk.shape

# %%
msk = msk.unsqueeze(2)

# %%
msk.shape

# %%
msk.shape[1]

# %%
def lookahead_mask(shape):
    # Mask out future entries by marking them with a 1.0
    mask = 1 - torch.tril(torch.ones(shape, shape))
    return mask

# %%
lookahead_mask(5)

# %%
x = torch.randn(64, 1, 5, 1, 20)

# %%
y = torch.randn(1, 5, 5)

# %%
reshaped_tensor = y.unsqueeze(0).unsqueeze(3).expand(64, 1, 5, 1, 5)


# %%
reshaped_tensor.shape

# %%
import torch

# Create float tensors
float_tensor1 = torch.tensor([1.5, 2.7, 3.9])
float_tensor2 = torch.tensor([2.3, 4.1, 3.2])

# Convert float tensors to integer tensors
int_tensor1 = float_tensor1.to(torch.int)
int_tensor2 = float_tensor2.to(torch.int)

# Perform bitwise AND operation
result_int = int_tensor1 & int_tensor2

# Convert the result back to float tensor
result_float = result_int.to(torch.float)

print("Result (integer):", result_int)
print("Result (float):", result_float)


# %%
int_tensor2

# %%
int_tensor1

# %%
class EncoderDecoder(nn.Module):
    def __init__(self, encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed  = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    def  encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)        

# %%
class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super().__init__()
        self.proj  =  nn.Linear(d_model,vocab)

    def forward(self,x):
        return log_softmax(self.proj(x),dim=-1)    

# %%
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# %%
class LayerNorm(nn.Module):
    def __init__(self, features,eps =1e-6):
        super().__init__()
        self.a_2  = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim= True)
        std = x.std(-1,keepdim= True)
        return self.a_2 * (x - mean)/(std +self.eps) +  self.b_2    

# %%
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model,d_ff,dropout = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def  forward(self,x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# %%
import math
class Embeddings(nn.Module):
    def __init__(self, d_model,vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    def forward(self,x):
        temp = self.lut(x) *  math.sqrt(self.d_model)
        return temp    


# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout ,max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000.0)/d_model) )
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:,:x.size(1)].requires_grad_(False)
        return self.dropout(x)    
        




# %%
pe1 = PositionalEncoding(20,0)
pe1.pe.shape

# %%
y = pe1(torch.zeros(1,100,20))

# %%
class Encoder(nn.Module):
    def __init__(self, layer,N):
        super().__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)    

# %%
import math
def attention(query,key,value,mask=None,dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn        

# %%
class MultiHeadedAttention(nn.Module):
    def __init__(self, h,d_model,dropout = 0.1):
        super().__init__()
        self. d_k = d_model//h
        self.h  = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask= None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query,key,value =[lin(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for lin,x in zip(self.linears,(query,key,value))] 
        x, self.attn = attention(query,key,value,mask,self.dropout)
        x = (x.transpose(1,2).contiguous().view(nbatches,-1,self.h * self.d_k))
        del query
        del key
        del value
        return self.linears[-1](x)

            

# %%
hds = 8
d_mdl = 512
at = MultiHeadedAttention(hds,d_mdl)

# %%
class SubLayerConnection(nn.Module):
    def __init__(self, size,dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    



# %%
class EncoderLayer(nn.Module):
    def __init__(self, size,self_attn,feed_forward,dropout):
        super().__init__()
        self.self_attn = self_attn
        self.sublayer = clones(SubLayerConnection(size,dropout),2)
        self.feed_forward = feed_forward
        self.size = size
    def forward(self,x,mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)


# %%
class DecoderLayer(nn.Module):
    def __init__(self, size,self_attn,src_attn,feed_forward,dropout):
        super().__init__()
        self.self_attn  = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.size = size
        self.sublayer = clones(SubLayerConnection(size,dropout),3)
    def forward ( self,x,memory,src_mask,tgt_mask):
            m = memory
            x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
            x =  self.sublayer[1](x,lambda x : self.src_attn(x,m,m,src_mask))
            return self.sublayer[2](x, self.feed_forward)

# %%
class Decoder(nn.Module):
    def __init__(self, layer,N):
        super().__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)    
            

# %%


# %%
class MyCustomObject:
    def __init__(self,value):
        self.value = value
        
        

# %%
def my_generator():
    for i in range(3):
         yield MyCustomObject(3)

# %%
for obj in my_generator():
    print(obj.value)

# %%
for x,y in enumerate(my_generator()):
    print(x)
    print(y)

# %%
import torch
import torch.nn as nn

# Define a simple model
model = nn.Linear(10, 5)

# Dummy input
input_data = torch.randn(1, 10)

# Forward pass
output = model(input_data)


# Dummy target
target = torch.randn(1, 5)

# Compute loss
criterion = nn.MSELoss()
loss = criterion(output, target)
#print("forward pass...")

#print(model.parameters().grad)

# Backward pass
loss.backward()
print("back pass..")
# Access gradients of parameters
for param in model.parameters():
    if param.grad is not None:
        print(param.grad)

# %%
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# Instantiate your model
model = MyModel()

# Access the parameters before computing the loss
for param in model.parameters():
    print(param)


# %%
inp = torch.randn(1,10)

# %%
op= model(inp)

# %%
target = torch.randn(1, 5)


# %%
criterion = nn.MSELoss()
loss = criterion(op, target)

# %%
loss.backward()

# %%
for param in model.parameters():
    if param.grad is not None:
        print(param.grad)

# %%
op.grad

# %%
for param in model.parameters():
    print(param)
    print(param.size())

# %%
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

# %%
data = torch.randint(1,11,size=(80,10))

# %%
data.shape

# %%
data[:5,:]

# %%
smask = (data!=0).unsqueeze(-2)

# %%
size  = smask.size(-1)

# %%
size

# %%
tgt = data[:,:-1]

# %%
tgt.shape

# %%
tgt_mask = (tgt!=0).unsqueeze(-2)

# %%
tgt_mask  = tgt_mask & subsequent_mask(tgt.size(-1))

# %%
src = torch.randn(2,2,3,3)
src

# %%
random_tensor = torch.rand(2,1,1,3)

# %%
random_tensor

# %%
msk = random_tensor > 0.5

# %%
msk

# %%
msk.shape

# %%
res = src.masked_fill(msk==0.3943,1e-9)

# %%
res

# %%
dm = 9
dt = torch.arange(0,dm,2)

# %%
dt

# %%
dt1 = dt.unsqueeze(1)

# %%
dt1

# %%
import math
dt2 = dt* (-(math.log(10000.0)/dm))

# %%
dt3  = dt2.unsqueeze(0)

# %%
psn = torch.arange(0,5000)

# %%
psn1 = psn.unsqueeze(1)

# %%
res = psn1*dt3

# %%
psn0 = psn.unsqueeze(0)

# %%
samp  = torch.zeros(4,3)

# %%
samp1 =  samp.unsqueeze(0)

# %%
x1 = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

# %%
p =torch.tensor([[[100, 200, 300, 400], [500, 600, 700, 800], [900, 1000, 1100, 1200]]])


# %%
c= copy.deepcopy

# %%
fd_frwrd= PositionwiseFeedForward(d_mdl,2048,0.1)

# %%
psn =PositionalEncoding(d_mdl,0.1)

# %%
enc_lyr = EncoderLayer(d_mdl,c(at),c(fd_frwrd),0.1)

# %%
enc = Encoder(enc_lyr,6)

# %%
dec_lyr = DecoderLayer(d_mdl,c(at),c(at),c(fd_frwrd),0.1)

# %%
dec = Decoder(dec_lyr,6)

# %%
s_embed = Embeddings(d_mdl,11)

# %%
src_s_embed = nn.Sequential(s_embed,c(psn))

# %%
t_embed = Embeddings(d_mdl,11)

# %%
tgt_embed = nn.Sequential(t_embed,c(psn))

# %%


# %%
gen = Generator(d_mdl,11)

# %%
mdl = EncoderDecoder(enc,dec,src_s_embed,tgt_embed,gen)

# %%
def make_model():
    hds = 8
    d_mdl = 512
    at = MultiHeadedAttention(hds,d_mdl)
    c= copy.deepcopy
    fd_frwrd= PositionwiseFeedForward(d_mdl,2048,0.1)
    psn =PositionalEncoding(d_mdl,0.1)
    enc_lyr = EncoderLayer(d_mdl,c(at),c(fd_frwrd),0.1)
    enc = Encoder(enc_lyr,2)
    dec_lyr = DecoderLayer(d_mdl,c(at),c(at),c(fd_frwrd),0.1)
    dec = Decoder(dec_lyr,2)
    s_embed = Embeddings(d_mdl,11)
    src_s_embed = nn.Sequential(s_embed,c(psn))
    t_embed = Embeddings(d_mdl,11)
    tgt_embed = nn.Sequential(t_embed,c(psn))
    gen = Generator(d_mdl,11)
    mdl = EncoderDecoder(enc,dec,src_s_embed,tgt_embed,gen)
    for p in mdl.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return mdl        




    
    
    
    

# %%
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

# %%
def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

# %%
V = 11
batch_size = 80
data_iter = data_gen(V, batch_size, 20)

# %%
for i, batch in enumerate(data_iter):
    s = batch.src
    print(i,batch.src.size)

# %%
t = batch.tgt

# %%
sm = batch.src_mask

# %%
tm = batch.tgt_mask

# %%
otpt = mdl.forward(s,t,sm,tm)

# %%
def greet(name):
    print(f"Hello {name}!")

# %%
def call(func,arg):
    func(arg) 

# %%
call(greet,"Alice")

# %%

class LabelSmoothing(nn.Module):
    def __init__(self,size,padding_idx,smoothing = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.paddiing_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None
    def forward(self,x,target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/(self.size  -  2))
        true_dist.scatter_(1,target.data.unsqueeze(1),self.confidence)
        true_dist[:,self.paddiing_idx] = 0
        mask = torch.nonzero(target.data == self.paddiing_idx)
        if mask.dim() > 0 :
            true_dist.index_fill_(0,mask.squeeze(),0.0)
        self.true_dist = true_dist
        return self.criterion(x,true_dist.clone().detach())    

# %%
class SimpleLossCompute:
    def __init__(self,generator,criterion):
        self.generator = generator
        self.criterion = criterion
    def __call__(self,x,y,norm):
        x = self.generator(x)
        sloss = (self.criterion(x.contiguous().view(-1,x.size(-1)),y.contiguous().view(-1))/norm)
        return sloss.data * norm,sloss
    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss    


# %%
optimizer =  torch.optim.Adam(mdl.parameters(),lr=0.5,betas=(0.9,0.98),eps=1e-9)

# %%
mdl.src_embed[0].d_model

# %%
smpl=src_s_embed[0].d_model

# %%
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

# %%
lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=mdl.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

# %%
V = 11
batch_size = 80
data_iter = data_gen(V, batch_size, 20)
crtrn = LabelSmoothing(V, padding_idx=0 ,smoothing =0.0)
#criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss_compute= SimpleLossCompute(gen, crtrn)
accum_iter = 1

# %%
class TrainState:
    step : int=0
    accum_step : int = 0
    samples : int = 0 
    tokens : int = 0

# %%
total_tokens = 0
total_loss = 0
tokens = 0 
n_accum = 0

# %%
train_state = TrainState()

# %%
def run_epoch():
    n_accum = 0
    total_tokens = 0
    total_loss = 0
    tokens = 0 
    n_accum = 0
   

    for i, batch in enumerate(data_iter):

        out = mdl.forward(batch.src,batch.tgt,batch.src_mask,batch.tgt_mask)
        #loss,loss_node = SimplelossCompute(gen,crtrn)
        loss,loss_node = loss_compute(out,batch.tgt_y,batch.ntokens)
        loss_node.backward()
        train_state.step +=1
        train_state.samples = batch.src.shape[0]
        train_state.tokens = batch.ntokens
        if i % accum_iter == 0 :


            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            n_accum += 1
            train_state.accum_step += 1
        lr_scheduler.step()
        total_loss += loss
        if i % 40 == 1:


            lr = optimizer.param_groups[0]["lr"]
            print(("Epoch step: %6d | Accumulation Step: %3d  | Loss : %6.2f" + "   | Learning Rate : %6.1e") % (i,n_accum,loss/batch.ntokens,lr)) 
            tokens = 0
        del loss
        del loss_node        


# %%
#run_epoch()

# %%
accum = 1
for i in range(20):
    print(i % accum)

# %%
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0 }]
        None  
    def step(self):
        None
    def zero_grad(self,set_to_none=False):
        None      

# %%
class DummyScheduler:
    def __init__(self):
        None
    def step(self):
        None

# %%

def run_epoch(data_iter,mdl,loss_compute,optimizer,scheduler,mode = "train",accum_iter =1,train_state=TrainState(),):
    n_accum = 0
    total_tokens = 0
    total_loss = 0
    tokens = 0 
    n_accum = 0
    
    for i, batch in enumerate(data_iter):
        out = mdl.forward(batch.src, batch.tgt, batch.src_mask,batch.tgt_mask)
        loss,loss_node = loss_compute(out,batch.tgt_y,batch.ntokens)
        
        if mode == "train" or mode== "train+log":
            loss_node.backward()
            train_state.step +=1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter==0:
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            print(("Epoch_step :%6d | Accumulation_step: %3d | Loss: %6.2f") % (i,n_accum,loss/batch.ntokens))
            tokens = 0
        del  loss
        del loss_node
    return total_loss/total_tokens,train_state    




# %%
def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size= V,padding_idx= 0 ,smoothing= 0.1)
    mdl  = make_model()
    optimizer = torch.optim.Adam(mdl.parameters(),lr = 0.5,betas= (0.9,0.98),eps = 1e-9)
    lr_scheduler = LambdaLR(

        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=mdl.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )
    batch_size = 80
    for epoch in range(20):
        mdl.train()
        run_epoch(data_gen(V,batch_size,20),mdl,SimpleLossCompute(mdl.generator,criterion),optimizer,lr_scheduler,mode = "train")
        mdl.eval()
        run_epoch(data_gen(V,batch_size,5),mdl,SimpleLossCompute(mdl.generator,criterion),DummyOptimizer(),DummyScheduler(),mode="eval")[0]
        mdl.eval()
    return mdl    
        
        

# %%
mdel=example_simple_model()
mdel

# %%
ys = torch.zeros(1,1)

# %%
ys.fill_(0)

# %%
ys

# %%
source = torch.tensor([[0,1,2,3,4,5,6,7,8,9]])

# %%
ys  = ys.type_as(source.data)

# %%
ys

# %%
y_size = ys.size(1)
y_size

# %%
ashape = (1,y_size,y_size)

# %%
tmp = torch.ones(ashape)
tmp

# %%
smsk = torch.triu(tmp,diagonal = 1)

# %%
smsk

# %%
smsk = smsk.type(torch.uint8)

# %%
smsk

# %%
rv = smsk==0

# %%
rv

# %%
tmask = rv

# %%
mem = mdel.encode(source,smsk)

# %%
out = mdel.decode(mem,smsk,ys,tmask.type_as(source.data))

# %%
prb = mdel.generator(out[:,-1])

# %%
_,nxt_wrd = torch.max(prb,dim=1)

# %%
nxt_wrd.data[0]

# %%
nxt_wrd = nxt_wrd.data[0]

# %%
ys = torch.cat([ys,torch.zeros(1,1).type_as(source.data).fill_(nxt_wrd)],dim=1)

# %%
def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")


    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
    return spacy_de,spacy_de    
            


# %%
class LanguageTokenizer:
    def __init__(self,spacy_model):
        self.spacy_model = spacy_model
    def tokenize(self,text):
        return [token.text for token in self.spacy_model(text)]    
        

# %%
class VocabularyBuilder:
    def __init__(self,spacy_de,spacy_en):
        self.tokenizer_de = LanguageTokenizer(spacy_de)
        self.tokenizer_en = LanguageTokenizer(spacy_en)

    @staticmethod
    def yield_tokens(data_iter, tokenizer,index):
        for from_to in data_iter:
            yield tokenizer(from_to[index])

    def build_vocabulary(self,spacy_de,spacy_en,train_data,val_data,test_data):
        print("Building German vocabulary...")
        vocab_src = build_vocab_from_iterator(self.yield_tokens(train_data + val_data + test_data,self.tokenizer_de.tokenize,index=0),min_freq=2,specials=["<s>", "</s>", "<blank>", "<unk>"]) 
        print("Building English vocabulary...")
        vocab_tgt = build_vocab_from_iterator(self.yield_tokens(train_data + val_data + test_data,self.tokenizer_en.tokenize,index=1),min_freq=2,specials=["<s>", "</s>", "<blank>", "<unk>"])

        vocab_src.set_default_index(vocab_src["<unk>"])
        vocab_tgt.set_default_index(vocab_tgt["<unk>"])
        return vocab_src,vocab_tgt

        

# %%
if is_interactive_notebook():
    spacy_de,spacy_en = show_example(load_tokenizers)
    


# %%
data_dir  = r'C:\Users\ADMIN\project\NLP\multi30k-dataset\data\task1\raw'
train_de_path = os.path.join(data_dir,'train.de.gz')
train_en_path = os.path.join(data_dir,'train.en.gz')
val_de_path = os.path.join(data_dir,'val.de.gz')
val_en_path = os.path.join(data_dir,'val.en.gz')
test_de_path = os.path.join(data_dir,'test_2016_flickr.de.gz')
test_en_path = os.path.join(data_dir,'test_2016_flickr.en.gz')


# %%
def read_data(src_path,tgt_path):
    with gzip.open(src_path,'rt',encoding='utf-8') as src_file, gzip.open(tgt_path,'rt',encoding='utf-8') as tgt_file:
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()
    return list(zip(src_lines,tgt_lines))    

# %%
train_data = read_data(train_de_path,train_en_path)
val_data= read_data(val_de_path,val_en_path)
test_data= read_data(test_de_path,test_en_path)

# %%
def load_vocab(spacy_de,spacy_en,train_data,val_data,test_data):
    if not exists("vocab.pt"):
        vb = VocabularyBuilder(spacy_de,spacy_en)
        vocab_src,vocab_tgt = vb.build_vocabulary(spacy_de,spacy_en,train_data,val_data,test_data)
        torch.save((vocab_src,vocab_tgt), "vocab.pt")
    else:
        with open("vocab.pt","rb") as f:
            vocab_src,vocab_tgt = torch.load(f)
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))       

    return vocab_src,vocab_tgt       

# %%
if is_interactive_notebook():
    spacy_de,spacy_en = show_example(load_tokenizers)
    vocab_src,vocab_tgt = show_example(load_vocab,args=[spacy_de,spacy_en,train_data,val_data,test_data])

# %%
for index, token in enumerate(vocab_src.get_itos()[:20]):
    print(f"Index:{index},Token:{token}")

# %%
for index,token in enumerate(vocab_tgt.get_itos()):
    print(f"Index:{index},Token:{token}")

# %%
vocab_src['ein']

# %%
vocab_src['vier']

# %%
vocab_src.get_itos()[3629]

# %%
src_sample = 'Ein weißer Hund mit einem Halsband läuft auf einer eingezäunten Rasenfläche umher.' 

# %%
src_sample_token = spacy_de.tokenizer(src_sample)

# %%
for token in src_sample_token:
    print(token.text)

# %%
src_sample_token[0]

# %%
vocab_src['Vier']

# %%
src_tokens = [tok.text for tok in spacy_de.tokenizer(src_sample)]

# %%
src_tokens

# %%
token_values = vocab_src(src_tokens)

# %%
token_values

# %%
for tokval in token_values:
    print(vocab_src.get_itos()[tokval])

# %%
vocab_src.get_itos()[5]

# %%
vocab_src.get_itos()[3]

# %%
processed_src = torch.cat([torch.tensor([0]),torch.tensor(token_values,dtype=torch.int64),torch.tensor([1]),],0,)

# %%
processed_src

# %%
src = pad(processed_src,(0,128 - len(processed_src)),value=2)

# %%
src

# %%
src.shape

# %%
class DataLoaders:
    def __init__(self, device,vocab_src,vocab_tgt,spacy_de,spacy_en,batch_size=12000,max_padding=128):
        self.device = device
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.spacy_de = spacy_de
        self.spacy_en = spacy_en
        self.batch_size = batch_size
        self.max_padding = max_padding
        self.tokenizer_de = LanguageTokenizer(spacy_de)
        self.tokenizer_en = LanguageTokenizer(spacy_en)
        torch.cuda.set_device(self.device)

    def tokenize_de(self,text):
        return self.tokenizer_de.tokenize(text) 
    
    def tokenize_en(self, text):
        return self.tokenizer_en.tokenize(text)
    def collate_fn(self,batch):
        return self.collate_batch(batch,self.tokenize_de,self.tokenize_en,self.vocab_src,self.vocab_tgt,max_padding=self.max_padding,pad_id =self.vocab_src.get_stoi()["<blank>"])
    def collate_batch(self,batch,src_pipeline,tgt_pipeline,src_vocab,tgt_vocab,max_padding=128,pad_id = 2):
        bs_id = torch.tensor([0]) # <s> token
        bs_id.to(self.device)
        eos_id = torch.tensor([1]) # </s> token
        eos_id.to(self.device)

        src_list = []
        tgt_list = []
        for _src,_tgt in batch:
            #processed_src = torch.cat([bs_id,torch.tensor(src_vocab(src_pipeline(_src)),dtype=torch.int64),eos_id],0,)
            processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    #device=device,
                ),
                eos_id,
            ],
            0,
        )
            processed_tgt = torch.cat([bs_id,torch.tensor(tgt_vocab(tgt_pipeline(_tgt)),dtype=torch.int64),eos_id],0,)
            src_list.append(pad(processed_src,(0,max_padding - len(processed_src)),value = pad_id))
            tgt_list.append(pad(processed_tgt,(0,max_padding - len(processed_tgt)),value = pad_id))
            
        src = torch.stack(src_list).to(self.device)
        tgt = torch.stack(tgt_list).to(self.device)
        return (src,tgt)

    def create_dataloaders(self):
        train_iter,valid_iter,test_iter = datasets.Multi30k(language_pair=("de", "en" ))
        train_dataloader = DataLoader(train_iter,batch_size=self.batch_size,shuffle=True,collate_fn=self.collate_fn )
        valid_dataloader = DataLoader(valid_iter,batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fn )
        return train_dataloader,valid_dataloader


        

# %%
def make_model_1(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# %%
model_1 = make_model_1(len(vocab_src),len(vocab_tgt),N=6)

# %%
model_1.cuda(0)

# %%
dev = 0
dataloader  = DataLoaders(dev,vocab_src,vocab_tgt,spacy_de,spacy_en,30 )

# %%
train_dataloader,valid_dataloader = dataloader.create_dataloaders()

# %%
#for batch in  train_dataloader:
    #src = batch.src.to(dev)
    #print(batch[0])

# %%
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

# %%
class ModelTrainer:
    def __init__(self, gpu,vocab_src,vocab_tgt,spacy_de,spacy_en,config ):
        self.device = gpu
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.spacy_de = spacy_de
        self.spacy_en = spacy_en
        self.config = config
        self.dataloader = DataLoaders(self.device,vocab_src,vocab_tgt,self.spacy_de,self.spacy_en,self.config["batch_size"],self.config["max_padding"])
        self.is_main_process = True
        #self.valid_dataloader =None
        train_dataloader, valid_dataloader = self.dataloader.create_dataloaders()
        self.valid_dataloader = valid_dataloader


    def train_worker(self):
        print(f"Train worker process using Gpu: {self.device} for training",flush = True)
        torch.cuda.set_device(self.device)
        pad_idx = self.vocab_tgt["<blank>"]
        d_model =512
        model = make_model_1(len(self.vocab_src), len(self.vocab_tgt), N=6)
        model.to(self.device)
        module = model
        
        
                                                             
        criterion = LabelSmoothing(len(self.vocab_tgt),pad_idx,0.1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),self.config["base_lr"],(0.9,0.98),1e-9)
        lr_scheduler = LambdaLR(optimizer,lr_lambda= lambda step:rate(step,d_model,factor=1,warmup=self.config["warmup"]),)
        train_state = TrainState()
        for epoch in range(self.config["num_epochs"]):
            model.train()
            print(f"[GPU{self.device}] Epoch{epoch} Training ===",flush=True)
            _,train_state = run_epoch((Batch(b[0],b[1],pad_idx) for b in train_dataloader),model,SimpleLossCompute(module.generator,criterion),optimizer,lr_scheduler,"train+log",accum_iter= self.config["accum_iter"],train_state=train_state)
            GPUtil.showUtilization()
            if self.is_main_process:
                file_path = "%s%.2d.pt" % (self.config["file_prefix"],epoch)
                torch.save(module.state_dict(), file_path)
            torch.cuda.empty_cache()
            print(f"[GPU{self.device}] Epoch{epoch} Validation ===",flush=True)
            model.eval()
            sloss = run_epoch((Batch(b[0],b[1],pad_idx) for b in valid_dataloader),model,SimpleLossCompute(module.generator,criterion),DummyOptimizer(),DummyScheduler(),mode ="eval")   
            print(sloss)
            torch.cuda.empty_cache()

        if self.is_main_process:
                
                file_path = "%sfinal.pt" % (self.config["file_prefix"])
                torch.save(module.state_dict(), file_path)


                                     


            

# %%
class TrainerConfig:
    def __init__(self):
        self.config = {
                       "batch_size": 32,
                       "num_epochs": 8,
                       "accum_iter" : 10,
                       "base_lr" : 1.0,
                       "max_padding": 72,
                       "warmup" : 3000,
                       "file_prefix" : "Multi30k_model_",



        }
        self.model_path = "Multi30k_model_final.pt"
        self.dev = 0
        self .model_trainer = ModelTrainer(self.dev,vocab_src,vocab_tgt,spacy_de,spacy_en,self.config)

    def load_model(self,vocab_src,vocab_tgt,spacy_de,spacy_en):
        if not exists (self.model_path):
            self.train_model()
        model = make_model_1(len(vocab_src),len(vocab_tgt),N=6)
        model.load_state_dict(torch.load(self.model_path))
        return model

    def train_model(self):
        self.model_trainer.train_worker()
              


    

# %%
trainer_config = TrainerConfig()
if is_interactive_notebook():
    trainer_config.load_model(vocab_src,vocab_tgt,spacy_de,spacy_en)

# %%
class ValidateModel(ModelTrainer):
    def __init__(self, device, vocab_src, vocab_tgt,spacy_de,spacy_en):
        super().__init__(device, vocab_src, vocab_tgt, spacy_de, spacy_en,trainer_config.config)
        self.model = None
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.spacy_de = spacy_de
        self.spacy_en = spacy_en
        self.device = device

        
    def load_model(self):
        self.model = make_model_1(len(self.vocab_src),len(self.vocab_tgt),N=6)
        self.model.load_state_dict(torch.load("Multi30k_model_final.pt"))
    def check_outputs(self,n_examples=15,pad_idx =2,eos_string = "</s>"):
        results = [()] * n_examples
        for idx in range(n_examples):
            print("\nExample %d ===============\n" % idx)
            b = next(iter(self.valid_dataloader))
            rb = Batch(b[0],b[1], pad_idx)
            #rb.src.to("cpu")
            src_tokens = [self.vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
            tgt_tokens = [self.vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]
            print("Source Text (Input)    :" + " ".join(src_tokens).replace("\n",""))
            print("Target Text (Ground Truth) :" + " ".join(tgt_tokens).replace("\n",""))
            model_out = greedy_decode(self.model,rb.src,rb.src_mask,72,0)[0]
            model_txt = (" ".join([self.vocab_tgt.get_itos()[x] for x in model_out if x !=pad_idx]).split(eos_string, 1)[0] + eos_string)
            print("Model Output          : " + model_txt.replace("\n",""))
            results[idx] = (rb,src_tokens,tgt_tokens,model_out,model_txt)
            
        return results
    
    def  run_model(self,n_examples=5):
        print("Preparing Data ...")
        print("Loading Trained Model ...")
        self.load_model()
        print("Checking Model Outputs ...")
        example_data = self.check_outputs()
        return example_data       
                

# %%
torch.cuda.current_device()

# %%
validated_model = ValidateModel(torch.device(torch.cuda.current_device()),vocab_src,vocab_tgt,spacy_de,spacy_en)

# %%
validated_model.run_model()

# %%
# Dummy data (replace with your actual data)
data_iter = [
    "This is the first sentence.",
    "And here's another one.",
    "Let's add a third sentence.",
]

# Import necessary libraries
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define a tokenizer (you can choose a different one based on your data)
tokenizer = get_tokenizer("basic_english")

# Function to yield tokens from data_iter
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build the vocab
vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Print some info about the vocab
print(f"Vocabulary size: {len(vocab)}")
print(f"Token for 'sentence': {vocab['sentence']}")
print(f"Token for 'unknown word': {vocab['unknown_word']}")


# %%
data_iter[0]

# %%
# Query the vocabulary for tokens in data_iter[0]
tokens_in_data_iter_0 = [vocab[token] for token in tokenizer(data_iter[0])]

# Print the integer values of tokens
print(tokens_in_data_iter_0)

# %%


# %%
idx =vocab.get_itos()

# %%
print(idx)

# %%
vocab_src(idx)

# %%
#tokenizer.

# %%
import random
class ToyDataset:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
    def shuffle_data(self):
       self.shuffled_indices = list(range(len(self.data)))
       random.shuffle(self.shuffled_indices)
       return self.shuffled_indices


       
       
    def unshuffle_data(self,indices):
        original_indices = [0] * len(indices)
        for i, shuffled_index in enumerate(self.shuffled_indices):
            original_indices[shuffled_index] = i
        return [original_indices[idx] for idx in indices]
        
        


# %%
def collate_fn(batch_data):
    return batch_data

# %%
toy_data = [
    {"input": "Hello", "target": "Bonjour"},
    {"input": "How are you?", "target": "Comment ça va?"},
    {"input": "I love programming.", "target": "J'aime programmer."},
    # Add more data samples here
]

# %%
toy_dataset = ToyDataset(toy_data)
batch_size = 2
shuffle = True

# %%
from torch.utils.data import DataLoader
import random

class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle, collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.index = 0
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices=self.dataset.shuffle_data()  # Assuming a shuffle_data method in the dataset

        #indices = list(range(len(self.dataset)))

        

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            batch = self.collate_fn(batch_data)
            yield batch

        if self.shuffle:
            indices = self.dataset.unshuffle_data(indices)    

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
# Usage example:
# train_dataloader = CustomDataLoader(train_iter_map, batch_size, shuffle=True, collate_fn)

# %%
trn_dataloader = CustomDataLoader(toy_dataset, batch_size, shuffle, collate_fn)

# %%
for batch in trn_dataloader:
    print(batch)
    lngth = len(batch)
    print(lngth)

# %%
ind = [0,1,2]

# %%
ind[0:2]

# %%
ind[2:4]

# %%
class DataLoaders:
    def __init__(self, device,vocab_src,vocab_tgt,spacy_de,spacy_en,batch_size=12000,max_padding=128):
        self.device = device
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.spacy_de = spacy_de
        self.spacy_en = spacy_en
        self.batch_size = batch_size
        self.max_padding = max_padding

    def tokenize(self,text,spacy_model):
        return [tok.text for tok in spacy_model(text)]
    def tokenize_en(self,text):
        return self.tokenize(text, self.spacy_en)
    def tokenize_de(self, text):
        return self.tokenize(text, self.spacy_de)
    
    

# %%
batch = [
    ("Hello, how are you?", "Bonjour, comment ça va?"),
    ("I love programming.", "J'aime programmer."),
    ("Let's go for a walk.", "Allons faire une promenade."),
]

# %%
for _src,_tgt in batch:
    print(_src)

# %%
spacy_de(_src)

# %%
tokens = [tok.text for tok in spacy_en(_src )]

# %%
tokens

# %%
values = vocab_src(tokens)

# %%
values

# %%
print(vocab_src.lookup_token(5))

# %%



