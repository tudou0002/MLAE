
#original ModelR
import pytorch_lightning as pl
import logging

from pytorch_lightning.utilities.types import STEP_OUTPUT
logger = logging.getLogger("GPT")
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import math
import tiktoken
import inspect
from dataclasses import dataclass
from ModuleR_patch_self_test3 import *
import torch
import torch.nn as nn
from torch.nn import functional as F
enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    patch_size: int = 16


class GPTReDo(pl.LightningModule): 
    
    def __init__(self, config_GPT, config_others):
        #config has to contain: 
        '''
        block_size
        vocab_size
        n_embd
        dropout
        n_layer
        bias
        '''
        super().__init__()
        self.save_hyperparameters() 
        assert config_GPT.vocab_size is not None
        assert config_GPT.block_size is not None
        
        self.config_GPT = config_GPT
        self.config = config_others
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config_GPT.vocab_size, config_GPT.n_embd),
            wpe = nn.Embedding(config_GPT.block_size + 32, config_GPT.n_embd),
            drop = nn.Dropout(config_GPT.dropout),
            h = nn.ModuleList([Block(config_GPT) for _ in range(config_GPT.n_layer)]),
            ln_f = LayerNorm(config_GPT.n_embd, bias=config_GPT.bias),
        ))
        
        
        self.token_self = nn.ModuleDict(dict(
            h = nn.ModuleList([DecoderBlock(config_GPT) for _ in range(config_GPT.n_layer)]),
            ln_f = LayerNorm(config_GPT.n_embd, bias=config_GPT.bias),
        ))
        #self.token_self = SelfAttnTokenLevelBlock(config_GPT)
        
        self.lm_head = nn.Linear(config_GPT.n_embd, config_GPT.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config_GPT.n_layer))
                
        # patch
        self.patch_size = config_GPT.patch_size
        print(f"patch_size: {self.patch_size}")
        self.down = nn.Linear(self.config_GPT.n_embd * self.patch_size, self.config_GPT.n_embd, bias=False)
        self.up = nn.Linear(self.config_GPT.n_embd, self.config_GPT.n_embd * self.patch_size, bias=False)
                
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        
    '''
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  #to do 
        return super().on_load_checkpoint(checkpoint)
    '''
        
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config_GPT.block_size
        self.config_GPT.block_size = block_size
        #crop the positional embedding to new block size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
                
                
    def forward(self, idx1, idx2, target1, target2):
        
        #print("entering forward")
        device = idx1.device
        b, t = idx1.size() 
        #assert t <= self.config_GPT.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config_GPT.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx1) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        #x = self.transformer.drop(tok_emb + pos_emb)
        x = self.transformer.drop(tok_emb )
        
        x2 = self.transformer.wte(idx2)
        
        
        # token to patch
        #bz, sq/pz , pz (16), dim
        #x_encoder = x[:, : self.config_GPT.block_size, :]
        #x = x[:, : self.config_GPT.block_size, :]
        x_patched = x.view(b, -1, self.patch_size, self.config_GPT.n_embd)
        #bz, sq/pz, pz * dim
        x_patched = x_patched.view(b, -1, self.patch_size * self.config_GPT.n_embd)
        # forward
        x_patched = self.down(x_patched) #bz, sq/pz (block nums), dim
        for block in self.transformer.h:
            x_patched = block(x_patched)
        x_patched2 = self.transformer.ln_f(x_patched) #bz, sq/pz, dim
        predict = self.up(x_patched2) #bz, sq/pz, dim * pz
        # patch to token
        #bz, sq/pz, pz, dim
        predict = predict.view(b, -1, self.patch_size, self.config_GPT.n_embd)
        
        autoencoder_logits = self.lm_head(predict.view(b, -1, self.config_GPT.n_embd))
        loss_auto = F.cross_entropy(autoencoder_logits.view(-1, autoencoder_logits.size(-1)), target1.view(-1), ignore_index=-1)
        
        # x_again = x[:, 15:, :].view(b, -1, self.patch_size, self.config_GPT.n_embd)
        # #predict = predict.view(b, -1, self.config_GPT.n_embd)
        
        
        for block in self.token_self.h:
            x2 = block(x2.view(b, -1, self.patch_size, self.config_GPT.n_embd), predict,x_patched )
        x2 = self.token_self.ln_f(x2) #bz, sq/pz, dim
        #predict = self.token_self(x2.view(b, -1, self.patch_size, self.config_GPT.n_embd) , predict)
        # #bz, sq, dim
        
        
        
        
        # #targets = idx
        
        
        if target2 is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x2)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target2.view(-1), ignore_index=-1)
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     #print("what shape", x.shape)
        #     logits = self.lm_head(predict[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None

        return autoencoder_logits, loss, loss_auto
        
        
    def training_step(self, batch, batch_idx):
        #print("entering training step")
        train_data1, train_data2, target1, target2 = batch[0], batch[1], batch[2], batch[3]
        #print("training shape", train_data.shape)
        auto_logits, loss, loss_auto = self.forward(train_data1, train_data2, target1, target2)
        #auto_logits: bz, sq, vocab_size
        assert loss != None
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        #self.log("train/CELoss", loss.detach(), **log_args) #log epoch_wise loss regradless of logger existance
        if(not self.logger==None):  # log step wise loss only on logger for visulization
                self.logger.experiment["train_CELoss_step"].append(loss.detach())
                self.logger.experiment["train_CELoss_auto_step"].append(loss_auto.detach())
        # if self.current_epoch < 0:
        #     return loss_auto
        # else:
        #     return loss+ loss_auto
        return  loss_auto 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #print("entering on_train_batch_end")
        
        #calculating the loss after clipping
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if(not self.logger==None): # step wise log
            self.logger.experiment["/grad_norm_after_clip"].append(total_norm)
            
            
    def on_after_backward(self) -> None:
        #print("entering after backward")
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
                
    
        
    def validation_step(self, batch, batch_idx) :
        #print("entering validation step")
        train_data1, train_data2, target1, target2 = batch[0], batch[1], batch[2], batch[3]
        
        #train_data1: bz, sq_len
        
        
        atuo_logits, loss , auto_loss= self.forward(train_data1, train_data2, target1, target2)
        assert loss != None
        
        first_logits = atuo_logits[0, :, :]
        decoded = first_logits.argmax(dim=-1)
        result = []
        
        result.append(decode(decoded.tolist()))
        
        
        first_input = target1[0, :]
        result.append(decode(first_input.tolist()))
        
        if(not self.logger==None):
                self.logger.experiment["val/case_study"].append("Epoch:  "+str(self.trainer.current_epoch)+ "decoded:" + str(result[0]))
                self.logger.experiment["val/case_study"].append("Epoch:  "+str(self.trainer.current_epoch)+"orignal: " + str(result[1]))
                
                
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log("valid/CELoss", loss.detach(), **log_args)
        self.log("valid/Auto_CELoss", auto_loss.detach(), **log_args)
        return auto_loss
    
    
    def predict_step(self, batch, *args): 
        idx, target = batch[0], batch[1]
        print("shape???", idx.shape)
        #print("entering predict")
        for _ in range(self.config["max_new_tokens"]):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config_GPT.block_size else idx[:, -self.config_GPT.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.config["temperature"]
            # optionally crop the logits to only the top k options
            if self.config["top_k"] is not None:
                v, _ = torch.topk(logits, min(self.config["top_k"], logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def configure_optimizers(self, device_type = "cuda") -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        print("entering optimizer step")
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': float(self.config["weight_decay"])},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available  and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=float(self.config["learning_rate"]), betas=(self.config["beta1"], self.config["beta1"]), **extra_args)
        lr_scheduler = MyScheduler(optimizer, self.config["warmup_iters"], self.config["max_epochs"], self.config["lr_decay_iters"], float(self.config["min_lr"]))
        print("exit optimizer")
        return [optimizer], [{"scheduler": lr_scheduler, "interval":"step"}]
    
    
    def on_before_optimizer_step(self, optimizer):
        #print("entering before optimizer")
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        #add by xiang, to flee from blowing, not working after exp
        '''
        if self.trainer.current_epoch >= 7:
            if total_norm >= 2.5:
                torch.nn.utils.clip_grad_norm(self.parameters(), 0.1)
        '''
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_before_clip"].append(total_norm)
            
    '''
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)  #to add the config files 
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
'''
class MyScheduler(torch.optim.lr_scheduler._LRScheduler):
    '''
    Lr scheduler with warmup and decay 
    '''
    def __init__(self, optimizer, warmup, max_iters, lr_decay_iters, min_lr):
        self.warmup_iters = warmup
        self.max_iters = max_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = float(min_lr)
        
        super().__init__(optimizer)
    def get_lr (self, ):
        it = self.last_epoch
        if it < self.warmup_iters:
            lr_factor = it / self.warmup_iters
            return [base_lr * lr_factor for base_lr in self.base_lrs]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return [self.min_lr for base_lr in self.base_lrs]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return [float(self.min_lr) + coeff * (base_lr - float(self.min_lr)) for base_lr in self.base_lrs]
        
    
        