import argparse, datetime, json, yaml, os,gc, time, itertools
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from main_model import CSDI_diabimmune
from dataset_diabimmune import get_dataset, read_data, diabimmune_Dataset
from utils import train, evaluate

import warnings
warnings.simplefilter('ignore', FutureWarning)


def sort_species(args, data, subjectID, timepoint, train_index, foldername):
    otu_column_name = data.columns.tolist()[0]
    
    subjectID_train = [subjectID[i] for i in train_index]
    colnames = ['.'.join(pair) for pair in itertools.product(subjectID_train, timepoint)]
    data = data[[otu_column_name]].join(data[colnames])
    
    data_cat = data.iloc[-(args.num_cat):, :]
    data = data.iloc[:(len(data) - args.num_cat), :]
    
    phylum_list = []
    for i in range(len(data)) :
        phylum_list.append(data[otu_column_name][i].split('|')[1])
    
    data["phylum"] = phylum_list 
    
    phylum_uniq_list = data["phylum"].unique().tolist()
    phylum_otu_count_list = []
    
    for i in range(len(phylum_uniq_list)) :
        tmp_phylum = data[data["phylum"] == phylum_uniq_list[i]]
        phylum_otu_count_list.append(len(tmp_phylum))
    
    phylum_otu_count_df = pd.DataFrame({'phylum' : phylum_uniq_list, 'otu_count' : phylum_otu_count_list})
    phylum_otu_count_df = phylum_otu_count_df.sort_values(by="otu_count", ascending = False)
    phylum_otu_count_df.reset_index(inplace = True, drop = True)
    
    final_data_df = pd.DataFrame()
    for i in range(len(phylum_otu_count_df)) : 
        tmp_phylum = data[data["phylum"] == phylum_otu_count_df["phylum"][i]]
        tmp_phylum.set_index(otu_column_name, inplace = True, drop= True)
        del tmp_phylum["phylum"]
        tmp_phylum = tmp_phylum.astype('float') # 2025/3/23 added
        colnames = tmp_phylum.columns.tolist()
        tmp_phylum = tmp_phylum.T
        tmp_phylum_corr = tmp_phylum.corr(method = 'spearman')
        tmp_phylum_otu_list = tmp_phylum_corr.index.tolist()
        tmp_phylum_otu_corr_list = []
        for otu in range(len(tmp_phylum_otu_list)) :
            tmp_phylum_otu_row_abs = tmp_phylum_corr.iloc[otu].abs()
            tmp_cumulative_corr = 1
            for j in range(len(tmp_phylum_otu_row_abs)) :
                if tmp_phylum_otu_row_abs[j] != 0.0 :
                    tmp_cumulative_corr *= tmp_phylum_otu_row_abs[j]
            tmp_phylum_otu_corr_list.append(tmp_cumulative_corr ** (1/len(tmp_phylum_otu_row_abs)))
        tmp_phylum_otu_corr_df = pd.DataFrame({'otu' : tmp_phylum_otu_list, 'corr' : tmp_phylum_otu_corr_list})
        tmp_phylum_otu_corr_df = tmp_phylum_otu_corr_df.sort_values(by = 'corr', ascending = False)
        tmp_data_based_on_corr = pd.merge(tmp_phylum_otu_corr_df, data, left_on = "otu", right_on = otu_column_name)
        del tmp_data_based_on_corr["corr"]
        del tmp_data_based_on_corr[otu_column_name]
        del tmp_data_based_on_corr["phylum"]
        tmp_data_based_on_corr["cluster"] = i
        final_data_df = pd.concat([final_data_df, tmp_data_based_on_corr], axis = 0)    

    attributes = list(final_data_df['otu']) + list(data_cat[otu_column_name])
    
    with open(f"{foldername}/attributes.txt", "w") as file:
        for item in attributes:
            file.write(item + "\n")

    return attributes


def get_cluster_size(attributes):
    cluster_size = pd.Series([x.split('|')[1] for x in attributes]).value_counts().values
    return cluster_size.tolist()


def main(args, config):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Read data
    data, subjectID, timepoint, taxons = read_data(args.dataset_path, args.num_cat) # taxons does not include category

    # To create dataset .pk file and to be used in kf.split
    dataset = diabimmune_Dataset(
        dataset_path=args.dataset_path, mask_path=args.mask_path, use_index_list=None, 
        missing_ratio=args.testmissingratio, seed=args.seed, attributes=None, num_cat=args.num_cat
    )

    mae_cv = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for _fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        # Create folder to store results
        foldername = f"./save/diabimmune_{current_time}_MR-{args.testmissingratio}_fold{_fold}"
        print('model folder:', foldername)
        os.makedirs(foldername, exist_ok=True)
        with open(f"{foldername}/config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Order bacteria based on training dataset
        attributes = sort_species(args, data, subjectID, timepoint, train_index, foldername) # attributes includes category

        # Get dataset
        train_dataset, valid_dataset, test_dataset = get_dataset(
            args=args, train_index=train_index, test_index=test_index, attributes=attributes
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=1)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=0)

        # Training & evaluation
        cluster_size = get_cluster_size(attributes[:-args.num_cat])
        cluster_size.append(args.num_cat * config["model"]["token_emb_dim"]) # 2025/3/23 added

        model = CSDI_diabimmune(config=config, device=args.device, 
                                target_dim=len(taxons) + args.num_cat * config["model"]["token_emb_dim"], 
                                cluster_size=cluster_size, num_cat=args.num_cat).to(args.device)

        if args.modelfolder == "":
            train(
                model,
                config["train"],
                train_loader,
                valid_loader=valid_loader,
                foldername=foldername,
            )
            output_path = foldername + "/model.pth"
            model.load_state_dict(torch.load(output_path)) 
        else:
            model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
        
        mae = evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, n_bacteria=len(taxons))
        mae_cv.append(mae)
        
        print("\nelapsed time: {0}".format(time.time() - time_start) + " [sec]")   
    
    print(f"Average MAE (SD): {np.mean(mae_cv)} ({np.std(mae_cv)})")    
    

if __name__ == '__main__':
    time_start = time.time()
    
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--taxon", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
 
    parser.add_argument("--dataset_path", type=str, default="../indata/DIABIMMUNE_16S/rel-species-table_clr_country.csv")
    parser.add_argument("--mask_path", type=str, default="../indata/DIABIMMUNE_16S/mask_0.1.csv")
    parser.add_argument("--num_cat", type=int, default=1)
    
    args = parser.parse_args()
    print(args)
    
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["test_missing_ratio"] = args.testmissingratio
    config["train"]["batch_size"] = args.batch_size
    config["train"]["epochs"] = args.epochs
    print(json.dumps(config, indent=4))
    
    main(args, config)

