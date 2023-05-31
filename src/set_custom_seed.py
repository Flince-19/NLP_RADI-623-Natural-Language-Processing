import random
import os
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False #Setting this to False might reduce performance.
#from AJ wanchana
##use like this        
#from my_custom_script.set_custom_seed import set_seed
#set_seed.set_seed(12)
