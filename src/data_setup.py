"""
Contains functions for setting up dataloder 
"""
import os
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple, Dict, List
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Subclass torch.utils.data.Dataset
class my_dataset(torch.utils.data.Dataset):
    
    # 2. Initialize with 
    def __init__(self, texts, labels, tokenizer, max_length, device)-> None:
        
        # 3. Create class attributes
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_texts = [tokenizer(text, padding='max_length', max_length = max_length, truncation=True, return_tensors="pt") for text in texts] #tokenized to raw text       
        self.labels = labels
        self.device = device

    # 4. Overwrite the __len__() method to get the amount of all samples
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.labels) 
    
    # 5. Overwrite the __getitem__() method 
    # It will take intex. It must return at least the sample and label.
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        "Returns one sample of data (tokenized text and label)"        
        text = self.tokenized_texts[index].to(device) #Send to device at this stage. Without this it fail at certain minibatch and I have no clue why. It just worked I swear.
        label = self.labels[index]

        return text, label #returns tokenized text (which for BERT is a dictionry of {'input_ids':tensor,'token_type_ids':tensor, attention_mnask:tensor}


#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0 #Somehow windows has problem with multi-process so > 0 does not works. IDK why.


def create_dataloaders(
    x_train: pd.core.series.Series, 
    y_train: pd.core.series.Series, 
    x_val: pd.core.series.Series,
    y_val: pd.core.series.Series,
    x_test: pd.core.series.Series,
    y_test: pd.core.series.Series,
    tokenizer,
    max_length,
    batch_size: int,
    device: device,
    num_workers: int=NUM_WORKERS):
    
  """Creates training and testing DataLoaders.

  Takes in panda series containing raw text (x) and label (y) from splitting of raw data with SK.learn.

  Args:
    x_train: panda series containing raw text.
    y_train: panda series containing encoded label
    x_val,x_test, y_val,y_test: Same as above but for val and test set.
    tokenizer: the tokenizer used for tokenizing the raw text.
    max_length: max sequence length for the tokenizer. If exceed, text will be truncated.
    device: device to sent the tensor to.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    train_data: custom dataset using my_dataset class
    val_data: custom dataset using my_dataset class
    test_data: custom dataset using my_dataset class
    train_dataloader: custom data loader generated from train_data
    val_dataloade: custom data loader generated from val_data
    test_dataloader: custom data loader generated from test_data

    Example usage:
        #Split train test first
        train_ratio = 0.8
        validation_ratio = 0.10
        test_ratio = 0.10

        df_ready = df_clean_1.copy()

        #Get the index of the tran, val and test set
        x_train, x_test, y_train, y_test = train_test_split(df_ready['transcription'], df_ready['label'], test_size=1 - train_ratio, stratify= df_ready['label'], random_state=50)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), stratify = y_test, random_state=50) 
        #get the text data from df_ready and reset the index to prepare the data which is to be inputt to the pytorch data loader later.
        x_train = df_ready.loc[x_train.index, 'transcription'].reset_index(drop = True)
        x_val =  df_ready.loc[x_val.index, 'transcription'].reset_index(drop = True)
        x_test =  df_ready.loc[x_test.index, 'transcription'].reset_index(drop = True)

        y_train = df_ready.loc[y_train.index, 'label'].reset_index(drop = True)
        y_val =  df_ready.loc[y_val.index, 'label'].reset_index(drop = True)
        y_test =  df_ready.loc[y_test.index, 'label'].reset_index(drop = True)    
        
        #Set to kenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #the cased model is note case-sensitive
        
        import data_setup
        train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(x_train=x_train,
                                                                                    y_train = y_train,                
                                                                                    x_val= x_val,
                                                                                    y_val = y_val,
                                                                                    x_test = x_test,
                                                                                    y_test = y_test,
                                                                                    tokenizer = tokenizer,
                                                                                    max_length = 512,                
                                                                                    batch_size=8,
                                                                                    device = device,
                                                                                    ) 

  """

  # Create dataset(s)
  train_data = my_dataset(texts= x_train, 
                          labels = y_train,
                          tokenizer = tokenizer,
                          max_length = max_length,
                          device = device)
  
  val_data = my_dataset(texts= x_val, 
                          labels = y_val,
                          tokenizer = tokenizer,
                          max_length = max_length,
                          device = device)            
                       
  test_data = my_dataset(texts= x_test, 
                          labels = y_test,
                          tokenizer = tokenizer,
                          max_length = max_length,
                          device = device)
                                     
                                     

  # Turn dataset into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      drop_last= True, 
      pin_memory=False)
  
  val_dataloader = DataLoader(
      val_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=False)  
  
  
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=False)
  

  return train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader
