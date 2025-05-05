import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import KFold
import time  
import itertools


time_start = time.time()

def input_data(profiles_filename, mask_filename):
    x_data = pd.read_csv(profiles_filename, index_col = 0) #Row: features, column:each sample
    
    # Added
    mask_data = pd.read_csv(mask_filename, index_col = 0).sort_index() #Row: sample, Column: timepoint
    mask_data.columns = mask_data.columns.astype('int')
    num_timepoint = len(mask_data.columns)
    mask_vector = mask_data.values
    mask_vector = mask_vector.reshape(-1, num_timepoint, 1)
    
    timepoints = mask_data.columns.tolist()
    
    x_data_tmp = x_data.T
    num_feature = len(x_data_tmp.columns)
    
    x_tmp = x_data_tmp.values
    x_tmp = x_tmp.reshape(-1, num_timepoint, num_feature)
    
    x_all = x_tmp.copy()
    mask_vector_all = mask_vector.copy()

    return timepoints, num_timepoint, num_feature, x_all, mask_vector_all


if __name__ == "__main__":
 
    profiles_filename = sys.argv[1]
    mask_filename = sys.argv[2]
    missing_ratio = float(sys.argv[3])

    # Read data
    timepoints, num_timepoint, num_feature, x_all, mask_vector_all = input_data(profiles_filename, mask_filename)

    mae_cv = []

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for _fold, (train_index, test_index) in enumerate(kf.split(x_all)):
        x = x_all[test_index]
        mask_vector = mask_vector_all[test_index]
        
        mask_vector_nan = mask_vector.astype('float')
        mask_vector_nan[mask_vector_nan == 0] = np.nan
        x_mask = x * mask_vector_nan
        x_mask_trans = x_mask.transpose(0, 2, 1)

        impute_vector = np.zeros(x_mask_trans.shape)
        for i in range(len(x_mask_trans)):
            for j in range(len(x_mask_trans[i])):
                impute_vector[i][j] = pd.Series(x_mask_trans[i][j]).interpolate(method = "linear", limit_direction="both")
        
        mae = np.abs(impute_vector.transpose(0, 2, 1) - x).sum() / ((1 - mask_vector).sum() * x_all.shape[2])
        print("MAE:", mae)
        mae_cv.append(mae)

        x_data = pd.read_csv(profiles_filename, index_col = 0)
        subjects = pd.Series([x.split('.')[0] for x in x_data.columns]).unique()
        impute_df = pd.DataFrame(impute_vector.transpose(0, 2, 1).reshape(-1, num_feature)).T
        impute_df.index = x_data.index
        impute_df.columns = list(itertools.chain.from_iterable([[x + '.' + str(y) for y in range(5)] for x in subjects[test_index]]))
        impute_df.to_csv("../outdata/imputed_Linear_clr_MR-" + str(missing_ratio) + "_fold" + str(_fold) + ".csv")


    print("Average MAE:", np.mean(mae_cv))




