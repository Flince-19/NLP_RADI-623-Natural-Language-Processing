
"""
Custom train and val loop
"""

import torch
from save_model import save_model
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                target_dir: str,
                figure_dir: str,
                model_name: str,
                epochs: int = 50,
                device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):


    """Full training/validate loop for a pytorch model and save model with the least loss

    Note: 
    - This function's logits is passed directly to the loss_function( for use with loss function such as crossentropy or BCEwithlogits).
    - If you want to use loss function which accept other input, you need to modify the training code
    - Does not utilize early stopping, only early save with val loss as criteria.
    - Automatically save model as .pth. It saves state_dict, epoch, best_val_loss and optimizer_state_dict.
    - Autoamtically save model's training result in pickle file in the same folder as the model.
    - The metrics used is f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score from scikitlearn.metrics.
    - Use Macro averaging for applicable metrics due to considering all class as equal. 
    - Depends on other custom function, save_model, found in utils_save_model.py
    - Automatically plot loss and metrics using matplotlib and save figure as SVG in figure_dir folder.
    - Device is prefebly setup in the beginning of the code in the device agnostic code cell
   
   Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    val_dataloader: A DataLoader instance for the model to be validated on.
    - The code unpack 2 variables from the data loader: X,y. X is output of bert tokenizer which has input_ids, attention_mask and token_type_ids
    - Since BERT expect input_ids and attention_mask, the input to the model needs to be X.input_ids, X.attention_mask
    epochs: An integer indicating how many epochs to train for.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    target_dir: target directory name to save the best model and train results with the least loss
    figure_dir: target directory name to save figure of plot
    model_name: Name of the model. Do not include .pth in the name.
    

  Returns:
    A dictionary of training and testing values for later plotting
    In the form: {train_loss_values: [...],
                  val_loss_values: [...],
                  etc...} 
    """
    
    print(f'Training on {device}.')
    # Create empty loss lists to track values
    
    train_loss_values = []
    val_loss_values = []
    epoch_count = []
    
    train_accuracy_values = []
    train_balanced_accuracy_values = []
    train_f1_values = []
    train_roc_auc_values = []
    train_precision_values = []
    train_recall_values = []    
    
    val_accuracy_values = []
    val_balanced_accuracy_values = []
    val_f1_values = []
    val_roc_auc_values = []
    val_precision_values = []
    val_recall_values = []   

    #initialize best lost
    val_loss_best = float('inf') #initialize best loss to inifite

    # Create training and testing loop
    for epoch in tqdm(range(epochs)): #use tqdm for progressbar
        print(f"Epoch: {epoch}\n-------") #print epoch
        ### Training
        train_loss = 0 #default train loss for that epoch.
        y_true_train, y_pred_train, y_prob_train = [], [], []  # Initialize empty lists for true and predicted labels
        y_true_val, y_pred_val, y_prob_val = [], [], []  # Initialize empty lists for true and predicted labels
        
        #This is just for tracking purpose
        
        # Add a loop to loop through training mini batches
        for batch, (X, y) in enumerate(train_dataloader): 
            #loop throuugh all minibatch in an epoch
            #We actually do not need to enumerate the dataloader and define batch
            #It can be just for X, y, path in train_dataloader: 
            #But batch can be used to keep track of progress of batch in each epoch so I just leave it in case you want to track the number of batch done
            
            X, y = X.to(device), y.to(device)
            
            model.train()  #set mode
            
            # 1. Forward pass
            y_logits = model(X.input_ids.squeeze(), X.attention_mask.squeeze()) #squeeze to get rid of extra dimension before inputting into BERT
            y_prob = torch.softmax(y_logits, dim = 1)
            y_pred = torch.argmax(y_prob, dim=1)
            
            #To get y_pred, we convert logits to probability using softmax in case of multiclass).
            #Then use argmax to get the label
            #This is not needed for training since we use BCE with LogitLoss which can directly use logits. However, we do it to calculate metrics.
             
            y_true_train.extend(y.tolist()) #accumulate true label
            y_prob_train.extend(y_prob.tolist())
            y_pred_train.extend(y_pred.tolist()) #accumulate pred label
            

            # 2. Calculate loss (per mini batch)
            #minibatch gradient descent calculate the loss and optimize the weight per mini batches 
            #as opposed to calculating and optimizing after all data has bee seen in an epoch
            loss = loss_fn(y_logits, y.type(torch.int64)) #Directly pass logits for Cross Entropy loss

            train_loss += loss # accumulatively add up the loss from ech minibatches fo tracking purpose

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step() #optimize network per mini batches batch
            
            # Print out how many samples or batch that have been seen every x iteration
            #sample
            # if batch % 1 == 0:
            #     print(f'Looked at {batch * len(X)}/{len(train_dataloader)} samples', end='\r')
            #batch
            print(f'Number of minibatch: {batch}/{len(train_dataloader)} batches', end='\r')



        
        #Find average train loss per minibatch
        train_loss /= len(train_dataloader)
        
        #Calculate metrics for that epoch using the accumulated label
        train_accuracy = accuracy_score(y_true_train, y_pred_train) 
        train_balanced_accuracy = balanced_accuracy_score(y_true_train, y_pred_train, adjusted = True) 
        train_f1_score = f1_score(y_true_train, y_pred_train, average = 'macro', zero_division = 'warn')   
        train_roc_auc_score = roc_auc_score(y_true_train, y_prob_train, average = 'macro', multi_class = 'ovo') #Documentation on Sklearn stated that ovo with macro si insensitive to class imbalance
        train_precision = precision_score(y_true_train, y_pred_train, average = 'macro', zero_division = 'warn')   
        train_recall = recall_score(y_true_train, y_pred_train, average = 'macro', zero_division = 'warn')

        
        ### val
        # Setup variables for accumulatively adding up loss and accuracy for that epoch
        val_loss = 0
        model.eval()
        
        with torch.inference_mode(): # context management
            for X, y in val_dataloader: # Loop through all mini batch in the val set. We don't use enumerate for this because
                # We don't care to track what test minibatch the progress is at.
                X, y = X.to(device), y.to(device)
                
                val_logits = model(X.input_ids.squeeze(), X.attention_mask.squeeze())
                val_prob = torch.softmax(val_logits, dim = 1)
                val_pred = torch.argmax(val_prob, dim=1)
            
            
                # 2. Calculate loss (accumatively)
                val_loss += loss_fn(val_logits, y.type(torch.int64)) # accumulatively add up the loss per mini batch for that epoch

                # 3. accumulate label for metric calculation
                y_true_val.extend(y.tolist()) #accumulate true label
                y_prob_val.extend(val_prob.tolist()) #accumualte predicted prob        
                y_pred_val.extend(val_pred.tolist()) #accumulate pred label


            # Calculations on val metrics need to happen inside torch.inference_mode()
            # Divide total val loss by length of val dataloader (per batch)
            val_loss /= len(val_dataloader)
            
            #Now you get the average test loss per minibatch for that epoch

            # For metric
            # Calculate metric based on accumulate true and pred label
            val_accuracy = accuracy_score(y_true_val, y_pred_val)
            val_balanced_accuracy = balanced_accuracy_score(y_true_val, y_pred_val, adjusted = True) 
            val_f1_score = f1_score(y_true_val, y_pred_val, average = 'macro', zero_division = 'warn')   
            val_roc_auc_score = roc_auc_score(y_true_val, y_prob_val, average = 'macro', multi_class = 'ovo') #Documentation on Sklearn stated that ovo with macro si insensitive to class imbalance
            val_precision = precision_score(y_true_val, y_pred_val, average = 'macro', zero_division = 'warn')   
            val_recall = recall_score(y_true_val, y_pred_val, average = 'macro', zero_division = 'warn')
            
            #for save model
            #check if the loss is less than the best val loss (using loss as threshold). You can change this to any metric.
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                best_epoch = epoch
                best_checkpoint = {
                    'best_epoch': best_epoch,
                    'best_val_loss': val_loss_best,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                
                
        #Printing of statistic every x epoch. Default to every epoch.    
        if epoch % 1 == 0:
            
            epoch_count.append(epoch) #append the number of epoch to be used as X axis to plot loss
            
            train_loss_values.append(train_loss.item()) #append data for train loss to plot
            
            val_loss_values.append(val_loss.item()) #append data for val loss to plot
            
            #append data for metric to plot
            train_accuracy_values.append(train_accuracy)
            train_balanced_accuracy_values.append(train_balanced_accuracy)
            train_f1_values.append(train_f1_score) 
            train_roc_auc_values.append(train_roc_auc_score)
            train_precision_values.append(train_precision)
            train_recall_values.append(train_recall)             
            
            
            val_accuracy_values.append(val_accuracy)
            val_balanced_accuracy_values.append(val_balanced_accuracy)
            val_f1_values.append(val_f1_score)
            val_roc_auc_values.append(val_roc_auc_score)
            val_precision_values.append(val_precision)
            val_recall_values.append(val_recall)  
                         
        ## Print out what's happening
        print(f"Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}")
        print(f"Train accuracy: {train_accuracy:.5f} | Train balanced accuracy: {train_balanced_accuracy:.5f} | Train F1 score: {train_f1_score:.5f} | Train ROC AUC score: {train_roc_auc_score:.5f}")    
        print(f"Validation accuracy: {val_accuracy:.5f} | Validation balanced accuracy: {val_balanced_accuracy:.5f} | Validation F1 score: {val_f1_score:.5f} | Val ROC AUC score: {val_roc_auc_score:.5f}\n")
        
    #Gather results into dictionary
    results = {'epochs': epoch_count,
               'best_epoch': best_epoch,
               'train_loss_values': train_loss_values,
               'val_loss_values': val_loss_values,
               'train_accuracy_values': train_accuracy_values,
               'val_accuracy_values': val_accuracy_values,
               'train_balanced_accuracy_values': train_balanced_accuracy_values,
               'val_balanced_accuracy_values': val_balanced_accuracy_values,
               'train_f1_values': train_f1_values,
               'val_f1_values': val_f1_values,               
               'train_roc_auc_values': train_roc_auc_values,
               'val_roc_auc_values': val_roc_auc_values,
               'train_precision_values': train_precision_values,
               'val_precision_values': val_precision_values,
               'train_recall_values': train_recall_values,
               'val_recall_values': val_recall_values       
    }    
    
    #Saving model with the best epoch.
    print(f'The best model is at epoch {best_epoch}')
    save_model(best_checkpoint, target_dir=target_dir, model_name=f'{model_name}_best_epoch_{best_epoch}.pth')
    
    #Saving the result dictionary in the same path which the model checkpoint is saved.
    results_file_path = f'{target_dir}\{model_name}_best_epoch_{best_epoch}_train_val_results.pkl'    
    print(f'Saving results dictionary as pickle file at ({results_file_path})')
    with open(results_file_path, 'wb') as file:  # Open the file in write mode
        # Write the dictionary to the pickle file
        pickle.dump(results, file)   
            
    #If want to open, do it like this:
    # Open the file in read mode and load the data
    # with open('..\model\BERT_LSTM_base_best_epoch_123_test_results.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)        
            
            
    #plot and save curves
    # Create target directory
    figure_dir_path = Path(figure_dir)
    figure_dir_path.mkdir(parents=True,
                        exist_ok=True)
    figure_save_path = figure_dir_path 
    
    print(f'Plotting and saving figure at: {figure_save_path}')
    # Plot the loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(results['epochs'], results['train_loss_values'], label="Train loss")
    plt.plot(results['epochs'], results['val_loss_values'], label="Val loss")
    plt.axvline(results['best_epoch'], linestyle='--', color='r',label=f'Best Epoch - {best_epoch} (lowest val loss)')
    plt.title(f'{model_name} - Training/Val loss curves')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f'{figure_save_path}\{model_name}_best_epoch_{best_epoch}_train_val_loss_curves.svg', format='svg')
    plt.show()

    # Plot the metric 
    plt.figure(figsize=(8, 4))
    plt.plot(results['epochs'], results['train_accuracy_values'], label="Train Accuracy")    
    plt.plot(results['epochs'], results['train_balanced_accuracy_values'], label="Train Balanced Accuracy")
    plt.plot(results['epochs'], results['train_f1_values'], label="Train F1 (Macro)")
    plt.plot(results['epochs'], results['val_accuracy_values'], label="Val Accuracy")    
    plt.plot(results['epochs'], results['val_balanced_accuracy_values'], label="Val Balanced Accuracy")
    plt.plot(results['epochs'], results['val_f1_values'], label="Val F1 (Macro)")
    plt.axvline(results['best_epoch'], linestyle='--', color='r',label=f'Best Epoch - {best_epoch} (lowest val loss)')
    plt.title(f'{model_name} - Train/Val metric 1')
    plt.ylabel("Score")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f'{figure_save_path}\{model_name}_best_epoch_{best_epoch}_train_val_metric_1.svg', format='svg')
    plt.show()

    # Plot the additional metric
    plt.figure(figsize=(8, 4))
    plt.plot(results['epochs'], results['train_roc_auc_values'], label="Train ROC AUC (Macro - OVO)")    
    plt.plot(results['epochs'], results['train_precision_values'], label="Train Precision (Macro)")
    plt.plot(results['epochs'], results['train_recall_values'], label="Train Recall (Macro)")
    plt.plot(results['epochs'], results['val_roc_auc_values'], label="Val ROC AUC (Macro - OVO)")    
    plt.plot(results['epochs'], results['val_precision_values'], label="Val Precision (Macro)")
    plt.plot(results['epochs'], results['val_recall_values'], label="Val Recall (Macro)")
    plt.axvline(results['best_epoch'], linestyle='--', color='r',label=f'Best Epoch: {best_epoch} (lowest val loss)')
    plt.title(f'{model_name} - Train/Val metric 2')
    plt.ylabel("Score")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f'{figure_save_path}\{model_name}_best_epoch_{best_epoch}_train_val_metric_2.svg', format='svg')
    plt.show()
    
    
    return results
