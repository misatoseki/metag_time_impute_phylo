import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import category_encoders as ce # 2024/6/28 added


def process_func(path: str, missing_ratio=0.1, attributes=None, eval_length=6, mask=None, metalist=""):
    data_ = pd.read_csv(path, index_col=0)
    data = data_.T[attributes]

    observed_values = np.array(data).reshape(int(data.shape[0]/eval_length), eval_length, data.shape[1]).astype("float32")
    observed_masks = ~np.isnan(observed_values)

    mask_array = np.array(mask)[:, :, np.newaxis]
    gt_masks = np.tile(mask_array, observed_masks.shape[2])

    metalist2 = metalist.copy()
    metalist2.insert(0, 'subjectID')
    metadata = pd.read_csv("../indata/metadata.csv", index_col=0)[metalist2].drop_duplicates()
    subjects = pd.DataFrame({'subjectID': [x.split('.')[0] for x in data_.columns]}).drop_duplicates() 

    encoder = ce.ordinal.OrdinalEncoder(cols=metalist)
    encoder.fit(metadata)
    new_df = encoder.transform(metadata)

    meta_data_array = np.array(pd.merge(subjects, new_df, on="subjectID")[metalist])[:, :, np.newaxis, np.newaxis]
    meta_data = np.tile(meta_data_array, [observed_masks.shape[1], observed_masks.shape[2]]) #[N,len(metalist),L,K]

    return observed_values, observed_masks, gt_masks, meta_data 


class diabimmune_Dataset(Dataset):
    def __init__(self, eval_length=6, use_index_list=None, missing_ratio=0.0, seed=0, attributes=None, inputdir="", metalist=""): 
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        self.meta_data = [] 
        path = (
            "./data/diabimmune_MR-" + str(missing_ratio) + ".pk"
        )

        dataset_path = "../indata/rel-species-table_clr.csv"
        mask = pd.read_csv("../indata/mask_" + str(missing_ratio) + ".csv", index_col=0)
        mask = mask.loc[np.sort(mask.index)]

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            (
                self.observed_values,
                self.observed_masks,
                self.gt_masks,
                self.meta_data #[N,len(metalist),L,K]
            ) = process_func(
                dataset_path,
                missing_ratio=missing_ratio,
                attributes=attributes,
                eval_length=eval_length,
                mask=mask,
                metalist=metalist 
            )

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, self.meta_data], f 
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, self.meta_data = pickle.load( 
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "meta_data": self.meta_data[index] 
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, inputdir="", attributes=None, foldername="",
                   train_index=None, test_index=None, metalist=""): 

    np.random.seed(seed)
    np.random.shuffle(train_index)
    num_train = (int)(len(train_index) * 0.9)
    tmp_index = train_index
    train_index = tmp_index[:num_train]
    valid_index = tmp_index[num_train:]

    dataset = diabimmune_Dataset(
        eval_length=5, use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, attributes=attributes, metalist=metalist 
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = diabimmune_Dataset(
        eval_length=5, use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, metalist=metalist
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = diabimmune_Dataset(
        eval_length=5, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, metalist=metalist
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader




