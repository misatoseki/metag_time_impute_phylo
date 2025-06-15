import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import category_encoders as ce


def process_func(data_, missing_ratio, attributes, eval_length, mask, num_cat):
    if attributes != None:
        data = data_.T[attributes] # Sort by attributes
    else:
        data = data_.T

    cat_list = np.arange(data.shape[1])[-num_cat:]
    cont_list = [i for i in range(0, data.shape[1] - len(cat_list))]

    # set encoder here
    encoder = ce.ordinal.OrdinalEncoder(cols=data.columns[-num_cat:])
    encoder.fit(data)
    new_df = encoder.transform(data)

    num_cate_list = []
    for index, col in enumerate(cat_list):
        num_cate_list.append(new_df.iloc[:, col].nunique())

    with open("./data/transformed_columns.pk", "wb") as f:
        pickle.dump([cont_list, num_cate_list], f)

    observed_values = np.array(new_df).reshape(int(new_df.shape[0]/eval_length), eval_length, new_df.shape[1]).astype("float32")
    observed_masks = ~np.isnan(observed_values)

    mask_array = np.array(mask)[:, :, np.newaxis]
    gt_masks = np.concatenate([np.tile(mask_array, observed_masks.shape[2]-1), 
                               np.ones([mask_array.shape[0], mask_array.shape[1], 1])], axis=2) # category is not masked

    return observed_values, observed_masks, gt_masks, cont_list


class diabimmune_Dataset(Dataset):
    def __init__(self, dataset_path, mask_path, use_index_list, missing_ratio, seed, attributes, num_cat):
        np.random.seed(seed)  # seed for ground truth choice

        self.missing_ratio = missing_ratio
        self.attributes = attributes

        data, subjectID, timepoint, taxons = read_data(dataset_path, num_cat)
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
            self.cont_cols,
        ) = process_func(
            data_=data,
            missing_ratio=self.missing_ratio,
            attributes=self.attributes,
            eval_length=self.eval_length,
            mask=self.mask,
            num_cat=num_cat
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
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes, num_cat=args.num_cat
    )
    valid_dataset = diabimmune_Dataset(
        dataset_path=args.dataset_path, mask_path=args.mask_path, use_index_list=valid_index, 
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes, num_cat=args.num_cat
    )
    test_dataset = diabimmune_Dataset(
        dataset_path=args.dataset_path, mask_path=args.mask_path, use_index_list=test_index, 
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=attributes, num_cat=args.num_cat
    )

    return train_dataset, valid_dataset, test_dataset


def read_data(dataset_path, num_cat):
    data = pd.read_csv(dataset_path)
    subjectID = sorted(list(set([x.split('.')[0] for x in data.iloc[:, 1:].columns])))
    timepoint = sorted(list(set([x.split('.')[1] for x in data.iloc[:, 1:].columns])))
    taxons = list(data.iloc[:(len(data)-num_cat), 0])
    
    return data, subjectID, timepoint, taxons


