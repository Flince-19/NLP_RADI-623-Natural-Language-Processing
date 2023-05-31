
import torch
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import timeit

"""
Functions to evaluation test set and create confusion matrix report
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

def test_set_prediction(model: torch.nn.Module,
                        test_dataloader: torch.utils.data.DataLoader,
                        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    """Used to evaluate model on test set dataloader

    Note: 
    - Assume multiclass problem, so use softmax in this case.

    Args:
    model: A PyTorch model to used.
    test_dataloader: test set of dataloader.
    - The code unpack 2 variables from the data loader: X,y. X is output of bert tokenizer which has input_ids, attention_mask and token_type_ids
    - Since BERT expect input_ids and attention_mask, the input to the model needs to be X.input_ids, X.attention_mask
    - figure_dir: target directory name to save figure of plot
    - result_dir: target directory name to save pickle file of result of test set


    Returns:
    y_true_tes: list of true label for each samples, 
    y_pred_test: list of predicted labels for each samples,
    y_pred_prob_test: list of predicted probability using sigmoid function for each samples
    """
    
    print(f'Inferencing on {device}.')
    
    #initialize empty list
    y_true_test, y_pred_test, y_prob_test = [], [], []

    model.eval()
    with torch.inference_mode(): 
        for X, y in tqdm(test_dataloader): #Loop through all minibatch in test_dataloader
            
            X, y = X.to(device), y.to(device)
            test_logits = model(X.input_ids.squeeze(), X.attention_mask.squeeze())
            test_prob = torch.softmax(test_logits, dim = 1)
            test_pred = torch.argmax(test_prob, dim=1)    
                  
            #To get y_pred, we convert logits to probability using soft max (multiclass)
            #Then use argmax to get the label.

            y_true_test.extend(y.tolist()) #accumulate true label
            y_pred_test.extend(test_pred.tolist()) #accumulate pred label
            y_prob_test.extend(test_prob.tolist()) #accumulate predicted prob
                
            
    return y_true_test, y_pred_test, y_prob_test


def calculate_metric(y_true: list,
                  y_pred: list,
                  y_prob: list = None):

    """Calculate metric

    Note: 
    - Output 6 metrics, accuracy, Balanced accuracy, F1 score, ROC AUC, precision, recall.
    - Average Precision (PRC AUC) is not suited for multiclass per SKlearn documentation

    Args:
    y_true: list of true label for each samples
    t_pred: list of predicted label for each samples
    y_prob: list of predicted probability for each samples. Does not use it by default
    
    Returns:
     accuracy: accuracy as calculated by SKlearn.metric
     balanced_accuracy: balanced accuracy as calculated by SKlearn.metric
     f1: f1 as calculated by SKlearn.metric
     roc_auc, precision, recall: as above
    """
    

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = 'macro')
    roc_auc = roc_auc_score(y_true, y_prob, average = 'macro', multi_class = 'ovo') #insensitive to class imbalance according to SK-learn documentation 
    precision = precision_score(y_true, y_pred, average = 'macro', zero_division = 'warn')   
    recall = recall_score(y_true, y_pred, average = 'macro', zero_division = 'warn')

    return accuracy, balanced_accuracy, f1, roc_auc, precision, recall
 

def test_set_report(model: torch.nn.Module,
                    model_file_name: str,
                    results_dir: str,
                    figure_dir:str,
                    test_dataloader: torch.utils.data.DataLoader,
                    class_map: dict = None,
                    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    """Create report of accuracy, Balanced accuracy, F1 score, ROC AUC, precision, recall, inference time and save those metric (including the actual prediction) to a pickle file in the form of dictionary to the specified directoy

    Note: 
    - PRC is restricted to binary classification.
    - Use Macro averaging for applicable metrics due to considering all class as equal.
    - Inference time is measure for all minibatch and is divided by the number of sample in the test dataloader to get the inference time per sample.
    - Inference time does not include model loading time and tokenization time as the test data has already been tokenized.

    Args:
    model: A PyTorch model to used.
    model_file_name: Name of the model file (to be used as the name in the results pickle file)    
    results_dir: Directory to save result pickle file in
    test_dataloader: test set of dataloader.
    - The code unpack 2 variables from the data loader: X,y. X is output of bert tokenizer which has input_ids, attention_mask and token_type_ids
    - Since BERT expect input_ids and attention_mask, the input to the model needs to be X.input_ids, X.attention_mask
    - figure_dir: target directory name to save figure of plot
    - result_dir: target directory name to save pickle file of result of test set
    class_map: a dictionary which map each "encoded" class to the name. Eg: {0:;benign', 1: 'malignant', 2:'Indeterminate'}
    device: device to do inference from
    
    Returns:

     accuracy: accuracy as calculated by SKlearn.metric
     balanced_accuracy: balanced accuracy as calculated by SKlearn.metric
     f1: f1 as calculated by SKlearn.metric
     roc_auc, precision, recall: as above
    """
    start_time = timeit.default_timer() #start timer
    y_true_test, y_pred_test, y_prob_test = test_set_prediction(model = model,
                                                                test_dataloader = test_dataloader,
                                                                device = device)
    end_time = timeit.default_timer()# stop timer
    inference_time_all = end_time - start_time   
    test_sample_number = len(test_dataloader.dataset)
    inference_time_one_sample = inference_time_all/test_sample_number
    
    accuracy, balanced_accuracy, f1, roc_auc, precision, recall = calculate_metric(y_true = y_true_test,
                                                                                   y_prob = y_prob_test,
                                                                                   y_pred = y_pred_test)
    #Print Classification Report
    if class_map:
        #If supplies class_map argument, create a new list from the dictionary.
        #It sort the label into a list accoring to the key. So value for key 0 will be the first item
        #Wich match how the target_names and display_labels argument expect to list of label to be (the first item in the list will be mapped to class 0, etc...)
        display_labels = [class_map[label] for label in sorted(class_map.keys())]
    else:
        display_labels = None


    confusion = confusion_matrix(y_true_test, y_pred_test, normalize='true')

    print(classification_report(y_true_test, y_pred_test, target_names= display_labels))
    
    #print('Confusion Matrix : \n', confusion)
    
    #plot and save matrix
    # Create target directory
    figure_dir_path = Path(figure_dir)
    figure_dir_path.mkdir(parents=True,
                        exist_ok=True)
    figure_save_path = figure_dir_path  
    
    print(f'Plotting and saving confusion matrix at: {figure_save_path}')
    disp = ConfusionMatrixDisplay(confusion, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(14,14))

    disp.plot(cmap='Blues', ax=ax, values_format='.2f', xticks_rotation='vertical', colorbar=False) #disable color bar 
    plt.title(f'{model_file_name} - Confusion Matrix')
    # Adding custom colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(disp.im_,  cax=cax)
      
    plt.savefig(f'{figure_save_path}\{model_file_name}_test_confusion.svg', format='svg')    
    plt.show()
    
    print('---------------------')        
    print(f'Inference time for all test sample ({device}): {inference_time_all:.2f} seconds')
    print(f'Number of all sample: {test_sample_number} samples')
    print(f'Inference time for one sample ({device}): {inference_time_one_sample:.2f} seconds')
    print('---------------------')      
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Balanced accuracy: {balanced_accuracy:.2f}')
    print(f'ROC AUC (Macro - OVO): {roc_auc:.2f}')
    print(f'Precision (Macro): {precision:.2f}')
    print(f'Recall (Macro): {recall:.2f}')
    print(f'F1 (Macro): {f1:.2f}')    
    print('---------------------')    
    #Gather results into dictionary
    results = {'model_name': model_file_name,
               'y_true': y_true_test,
               'y_pred': y_pred_test,
               'y_prob': y_prob_test,
               'test_accuracy': accuracy,
               'test_balanced_accuracy': balanced_accuracy,
               'test_f1': f1,         
               'test_roc_auc': roc_auc,
               'test_precision': precision,
               'test_recall': recall,
               'inference_time_all': inference_time_all,
               'test_sample_number': test_sample_number,
               'inference_time_one_sample': inference_time_one_sample,
               'device': device}    
    #save results as pickle file in the specified directory
    
    # Create target directory
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True,
                        exist_ok=True)
    results_save_path = results_dir_path 
    
    #Saving the result dictionary
    results_file_path = f'{results_save_path}\{model_file_name}_test_results.pkl'    
    print(f'Saving results dictionary as pickle file at ({results_file_path})')
    with open(results_file_path, 'wb') as file:  # Open the file in write mode
        # Write the dictionary to the pickle file
        pickle.dump(results, file)   
        
    #If want to open, do it like this:
    # Open the file in read mode and load the data
    # with open('..\model\BERT_LSTM_base_best_epoch_123_test_results.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)    
        
    return results
    
    

    
    
