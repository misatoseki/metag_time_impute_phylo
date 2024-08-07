import numpy as np 
import torch
import pickle
import pandas as pd
from sklearn.model_selection import KFold
from dataset_diabimmune import diabimmune_Dataset
import itertools


def make_prof(foldername="", nsample=100, missing_ratio=None, inputdir="../indata", imp="", seed=None):
    # Read imputed profile
    path = foldername + "/generated_outputs_nsample" + str(nsample) + ".pk"   
    
    with open(path, 'rb') as f:
        samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load( f)
    
    # Set column label
    with open(inputdir + "/attribute", "r") as f:
        attributes = f.read().split('\n')
        attributes.remove('') 
   
    # Create imputed dataset
    imputed_tensor = samples.median(dim=1).values
    
    # Alternate values if observed
    prof_tensor = imputed_tensor * all_evalpoint + all_target * (1 - all_evalpoint)
    profile = pd.DataFrame(prof_tensor.reshape(-1, all_target.shape[2]).to('cpu').numpy())

    # if checking MAE
    print("MAE:", np.abs((imputed_tensor * all_evalpoint - all_target * all_evalpoint).cpu()).sum() / all_evalpoint.sum())


    cols = pd.read_csv(inputdir + "/rel-species-table_clr.csv", index_col=0).columns
    subjects = pd.DataFrame({'subj': [x.split('.')[0] for x in cols]}).drop_duplicates()['subj'].values

    dataset = diabimmune_Dataset(eval_length=5, missing_ratio=missing_ratio, seed=1, attributes=attributes, inputdir=inputdir)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for _fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        break
    
    # Merge index names
    profile.columns = attributes
    profile.index = list(itertools.chain.from_iterable([[x + '.' + str(y) for y in range(5)] for x in subjects[test_index]])) 
    
    profile.T.to_csv("../outdata/imputed_CSDI_modified_clr_MR-" + str(missing_ratio) + "_fold0.csv")


if __name__=='__main__':
    make_prof(foldername="./save/diabimmune_20240627_021641_MR-0.1_fold0", missing_ratio=0.1)


