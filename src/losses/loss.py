import torch
import torch.nn.functional as F

def loss_fn(y_true , y_pred ) : 
    y_pred = y_pred.view(-1 ,y_pred.shape[-1]) 
    y_true  = y_true.view(-1 , y_true.shape[-1])
    log_probs  = torch.log_softmax(y_pred , dim =-1)
    loss = -(y_true * log_probs).sum(dim=-1).mean()
    return loss