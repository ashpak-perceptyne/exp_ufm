import torch 

class Early_stoping : 
    """
    This callback monitors the validation loss and stops the training process
    when the loss has not improved by more than `min_delta` for a number of
    epochs equal to `patience`.

    """
    def __init__(self, min_delta : float , patiance : int) -> None : 

        self.patiance = patiance
        self.min_delta = min_delta 
        self.best_loss :float = float('inf') 
        self.wait = 0 
        
    def on_epoch_end(self , val_loss , epoch) : 
        if val_loss < self.best_loss - self.min_delta : 
            self.best_loss  = val_loss 
            self.wait = 0 
        else : 
            self.wait+=1 
        
        if self.wait > self.patiance : 
            print(f'Early stoping at {epoch}') 
            return True 

        return False 
