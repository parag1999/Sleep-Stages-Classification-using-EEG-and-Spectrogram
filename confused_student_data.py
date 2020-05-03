import os
import pandas as pd
import numpy as np

def csv_to_dict():
    path = os.getcwd()+"/EEG_data.csv"
    data_df = pd.read_csv(path)
    data_dict  ={}
    
    for index, row in data_df.iterrows(): 
        id = (int(row["SubjectID"]), int(row["VideoID"]))
        if  id in data_dict:
            data_dict[id]["raw"].append(row["Raw"])
            data_dict[id]["alpha_1"].append(row["Alpha1"])
            data_dict[id]["alpha_2"].append(row["Alpha2"])
            data_dict[id]["beta_1"].append(row["Beta1"])
            data_dict[id]["beta_2"].append(row["Beta2"])
            data_dict[id]["gamma_1"].append(row["Gamma1"])
            data_dict[id]["gamma_2"].append(row["Gamma2"])
            data_dict[id]["delta"].append(row["Delta"])
            data_dict[id]["theta"].append(row["Theta"])
        else:
            data_dict[id] = {
                    "raw":[row["Raw"]],
                    "label":int(row["user-definedlabeln"]),
                    "alpha_1":[row["Alpha1"]],
                    "alpha_2":[row["Alpha2"]],
                    "beta_1":[row["Beta1"]],
                    "beta_2":[row["Beta2"]],
                    "gamma_1":[row["Gamma1"]],
                    "gamma_2":[row["Gamma2"]],
                    "delta":[row["Delta"]],
                    "theta":[row["Theta"]]
                    }
        
    for uid in data_dict:
        print(uid)
        data_dict[uid]["raw"] = np.array(data_dict[uid]["raw"])
        data_dict[uid]["alpha_1"] = np.array(data_dict[uid]["alpha_1"])
        data_dict[uid]["alpha_2"] = np.array(data_dict[uid]["alpha_2"])
        data_dict[uid]["beta_1"] = np.array(data_dict[uid]["beta_1"])
        data_dict[uid]["beta_2"] = np.array(data_dict[uid]["beta_2"])
        data_dict[uid]["gamma_1"] = np.array(data_dict[uid]["gamma_1"])
        data_dict[uid]["gamma_2"] = np.array(data_dict[uid]["gamma_2"])
        data_dict[uid]["delta"] = np.array(data_dict[uid]["delta"])
        data_dict[uid]["theta"] = np.array(data_dict[uid]["theta"])
                
    return data_dict

if __name__ == "__main__":
    a = csv_to_dict()
    #Open it in spyder and view in variable explorer
    #Use it for simple time series data experiments
    