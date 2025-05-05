import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_func(data_, missing_ratio, attributes, eval_length, mask):
    if attributes != None:
        data = data_.T[attributes] # Sort by attributes
    else:
        data = data_.T

    observed_values = np.array(data).reshape(int(data.shape[0]/eval_length), eval_length, data.shape[1]).astype("float32")
    observed_masks = ~np.isnan(observed_values)

    mask_array = np.array(mask)[:, :, np.newaxis]
    gt_masks = np.tile(mask_array, observed_masks.shape[2])

    return observed_values, observed_masks, gt_masks


class diabimmune_Dataset(Dataset):
    def __init__(self, dataset_path, mask_path, use_index_list, missing_ratio, seed, attributes):
        np.random.seed(seed)  # seed for ground truth choice

        self.missing_ratio = missing_ratio
        self.attributes = attributes

        data, subjectID, timepoint, taxons = read_data(dataset_path)
        otu_column_name = data.columns.tolist()[0]
        data = data.set_index(otu_column_name)

        self.subjectID = subjectID
        self.eval_length = len(timepoint)

        mask = pd.read_csv(mask_path, index_col=0)
        mask = mask.loc[np.sort(mask.index)]
        self.mask = mask

        (
            self.observed_values,
            self.observed_masks,
            self.gt_masks,
        ) = process_func(
            data_=data,
            missing_ratio=self.missing_ratio,
            attributes=self.attributes,
            eval_length=self.eval_length,
            mask=self.mask
        )

        data_save_path = ("./data/diabimmune_MR-" + str(missing_ratio) + ".pk")
        if os.path.isfile(data_save_path) == False:  # if datasetfile is none, create
            with open(data_save_path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)

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
            "subjectID": self.subjectID[index],
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataset(args, train_index, test_index, attributes):
    # Create validation data index
    np.random.seed(args.seed)
    np.random.shuffle(train_index)
    num_train = (int)(len(train_index) * 0.9)
    tmp_index = train_index
    train_index = tmp_index[:num_train]
    valid_index = tmp_index[num_train:]

    # Create datasets
    train_dataset = diabimmune_Dataset(
        dataset_path=args.dataset_path, mask_path=args.mask_path, use_index_list=train_index, 
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes
    )
    valid_dataset = diabimmune_Dataset(
        dataset_path=args.dataset_path, mask_path=args.mask_path, use_index_list=valid_index, 
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes
    )
    test_dataset = diabimmune_Dataset(
        dataset_path=args.dataset_path, mask_path=args.mask_path, use_index_list=test_index, 
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes
    )

    return train_dataset, valid_dataset, test_dataset


def read_data(dataset_path):
    data = pd.read_csv(dataset_path)
    subjectID = sorted(list(set([x.split('.')[0] for x in data.iloc[:, 1:].columns])))
    timepoint = sorted(list(set([x.split('.')[1] for x in data.iloc[:, 1:].columns])))
    taxons = list(data.iloc[:, 0])
    
    return data, subjectID, timepoint, taxons


