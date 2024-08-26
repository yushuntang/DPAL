"""
Copyright to DPAL Authors, ECCV 2024
built upon on Tent and SAR code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math, json
import numpy as np


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class DPAL(nn.Module):
    """DPAL online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once DPALed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, args, optimizer, ad_optimizer, steps=1, episodic=False, margin_e0=1.0*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.ad_optimizer = ad_optimizer
        self.steps = steps
        assert steps > 0, "DPAL requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        # self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.model_state, self.optimizer_state, self.ad_optimizer_state = copy_model_and_optimizer(self.model, self.optimizer, self.ad_optimizer)
        

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            self.attention_weights = None
            outputs, ema, reset_flag = self.forward_and_adapt_dpal(x, self.model, self.optimizer, self.ad_optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            # stop
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.ad_optimizer,
                                 self.model_state, self.optimizer_state, self.ad_optimizer_state)
        self.ema = None

    def get_attention_weights(self, module, input, output):
        if self.attention_weights is None:
            self.attention_weights = output.mean(0).mean(0)[0,2:].unsqueeze(0)
        else:
            self.attention_weights = torch.cat((self.attention_weights, output.mean(0).mean(0)[0,2:].unsqueeze(0)), dim=0)

    def get_attention_weights_before_softmax(self, module, input, output):
        B, N, C = input[0].shape
        qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * (64 ** -0.5)

        if self.attention_weights_before_softmax is None:
            self.attention_weights_before_softmax = attn.mean(0).mean(0)[0,2:].unsqueeze(0)
        else:
            self.attention_weights_before_softmax = torch.cat((self.attention_weights_before_softmax, attn.mean(0).mean(0)[0,2:].unsqueeze(0)), dim=0)


    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    
    @torch.jit.script
    def entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x * x.log()).sum(1)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_dpal(self, x, model, optimizer, ad_optimizer, margin, reset_constant, ema):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        optimizer.zero_grad()
        ad_optimizer.zero_grad()
        # forward
        outputs, noises_output = model(x)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        similarity_loss = torch.stack([-nn.functional.cosine_similarity(noise.unsqueeze(1), noise.unsqueeze(0), dim=2).mean() for noise in noises_output]).mean()
        entropys = self.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < margin)
        entropys = entropys[filter_ids_1]
        loss = (entropys.mean(0) + (len(filter_ids_1[0])/self.args.test_batch_size) * similarity_loss)
        loss.backward()

        ad_optimizer.step()

        optimizer.first_step(zero_grad=True) 
        outputs, noises_output = model(x)
        entropys2 = self.softmax_entropy(outputs)
        entropys2 = entropys2[filter_ids_1]  
        filter_ids_2 = torch.where(entropys2 < margin)  
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        
        optimizer.second_step(zero_grad=True)
        

        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        return outputs, ema, reset_flag

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_tent(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model(x)
        # adapt
        loss = self.softmax_entropy(outputs).mean(0) - self.args.lambda_entropy*self.entropy(self.attention_weights).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs



def collect_params(model, args):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        # if 'blocks.18' in nm:
        #     continue
        # if 'blocks.19' in nm:
        #     continue
        # if 'blocks.20' in nm:
        #     continue
        if 'blocks.21' in nm:
            continue
        if 'blocks.22' in nm:
            continue
        if 'blocks.23' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    p.requires_grad_(True)
                    params += [{'params': p, 'lr': args.norm_lr}]
                    names.append(f"{nm}.{np}")
            
            
        for np, p in m.named_parameters():
            if 'prompt_embedding' in np:
                p.requires_grad_(True)
                params += [{'params': p, 'lr': args.prompt_lr}]
                names.append(f"{nm}.{np}")

    return params, names

def collect_prodictor_params(model, args):
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if 'predictor' in np:
                p.requires_grad_(True)
                # params.append(p)
                params += [{'params': p, 'lr': args.predictor_lr}]
                names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer, ad_optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    ad_optimizer_state = deepcopy(ad_optimizer.state_dict())
    return model_state, optimizer_state, ad_optimizer_state


def load_model_and_optimizer(model, optimizer, ad_optimizer, model_state, optimizer_state, ad_optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    ad_optimizer.load_state_dict(ad_optimizer_state)


def configure_model(model):
    """Configure model for use with DPAL."""
    # train mode, because DPAL optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DPAL updates
    model.requires_grad_(False)
    # configure norm for DPAL updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with DPAL."""
    is_training = model.training
    assert is_training, "DPAL needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "DPAL needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "DPAL should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "DPAL needs normalization layer parameters for its optimization"
