
from contextlib import nullcontext
from model import ModelArgs
import torch
from model import Transformer
from tokenizer import Tokenizer


checkpoint = "stories15M.pt"
device = 'cpu'
max_new_tokens = 100
temperature = 1.0
top_k = 300
num_samples = 1
ctx = nullcontext()

checkpoint_dict = torch.load(checkpoint, map_location = 'cpu')
gptconf = ModelArgs(**checkpoint_dict['model_args'])

model = Transformer(gptconf)
state_dict = checkpoint_dict['model']

model.load_state_dict(state_dict, strict = False)
model.eval()
model.to('cpu')

vacab_source = checkpoint_dict.get('vacab_source', 'llama2')
vovab_size = gptconf.vocab_size

seed = 1337
torch.manual_seed(seed)

enc = Tokenizer(tokenizer_model=None)
start = ''
start_ids = enc.encode(start, bos = True, eos = False)

x = (torch.tensor(start_ids, dtype = torch.long, device = device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature = temperature, top_k = top_k)
            print(y)
            y[0].tolist()
            print(enc.decode(y[0].tolist()))

