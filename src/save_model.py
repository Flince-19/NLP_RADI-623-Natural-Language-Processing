"""
For model saving.
"""
import torch
from pathlib import Path

def save_model(object,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  #Note that what is saved depends on what you defined the object in the training,val loop!
  # For example, this code in the training loop save epoch, val loss, state_dict and optimizer
  # best_checkpoint = {
  #     'best_epoch': best_epoch,
  #     'best_val_loss': val_loss_best,
  #     'model_state_dict': model.state_dict(),
  #     'optimizer_state_dict': optimizer.state_dict()
  # }
  
  print(f"Saving checkpoint at ({model_save_path})")
  
  #save state state_dict and optimizer

  torch.save(obj=object,
             f=model_save_path)
 

#Use it like this
# Import utils.py
# from going_modular import utils

# # Save a model to file
# save_model(object=...
#            target_dir=...,
#            model_name=...)
