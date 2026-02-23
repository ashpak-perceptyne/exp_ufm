import torch
import torch.nn as nn

def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    loss: float, 
    ckpt_path: str = 'model_probe.pth'
) -> None: 
    """
    Saves the trainable projection layers, optimizer state, and experiment metadata.
    Avoids saving the frozen DINOv2 backbone.

    Args:
        model (nn.Module): The hybrid_model containing the .probs ModuleDict.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current training epoch.
        loss (float): The current validation or training loss.
        ckpt_path (str): The file path to save the checkpoint.
    """
    
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.probs.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} at epoch {epoch} with loss {loss:.4f}")