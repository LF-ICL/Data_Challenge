import warnings
from tqdm import TqdmExperimentalWarning
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from traffic.core import Traffic
from traffic.data import airports

# print(airports["EHAM"].latlon)

challenge_set =pd.read_csv("challenge_set_addParq.csv")
final_submission_set =pd.read_csv("final_submission_set_addParq.csv")

ap_list = challenge_set['adep'].unique().tolist()
ap_list.extend(challenge_set['ades'].unique().tolist())
ap_list.extend(final_submission_set['adep'].unique().tolist())
ap_list.extend(final_submission_set['ades'].unique().tolist())
ap_list = list(set(ap_list))
print(ap_list)
ap_dic = {}
aplat_dic = {}
aplon_dic = {}
apName_dic = {}
for i in range(len(ap_list)):
    try:
        ap_dic[ap_list[i]] = airports[ap_list[i]].latlon
        aplat_dic[ap_list[i]] = airports[ap_list[i]].latlon[0]
        aplon_dic[ap_list[i]] = airports[ap_list[i]].latlon[1]
        apName_dic[ap_list[i]] = airports[ap_list[i]].iata
    except:
        print("Unknown airport "+ap_list[i]+" in current database")

# print(aplat_dic)
challenge_set["ades_lat"]  = challenge_set["ades"].map(aplat_dic)
challenge_set["ades_lon"]  = challenge_set["ades"].map(aplon_dic)
challenge_set["ades_latlon"] = challenge_set["ades"].map(ap_dic)
challenge_set["ades_name"] = challenge_set["ades"].map(apName_dic)
challenge_set["adep_lat"]  = challenge_set["adep"].map(aplat_dic)
challenge_set["adep_lon"]  = challenge_set["adep"].map(aplon_dic)
challenge_set["adep_latlon"] = challenge_set["adep"].map(ap_dic)
challenge_set["adep_name"] = challenge_set["adep"].map(apName_dic)

final_submission_set["ades_lat"]  = final_submission_set["ades"].map(aplat_dic)
final_submission_set["ades_lon"]  = final_submission_set["ades"].map(aplon_dic)
final_submission_set["ades_latlon"] = final_submission_set["ades"].map(ap_dic)
final_submission_set["ades_name"] = final_submission_set["ades"].map(apName_dic)
final_submission_set["adep_lat"]  = final_submission_set["adep"].map(aplat_dic)
final_submission_set["adep_lon"]  = final_submission_set["adep"].map(aplon_dic)
final_submission_set["adep_latlon"] = final_submission_set["adep"].map(ap_dic)
final_submission_set["adep_name"] = final_submission_set["adep"].map(apName_dic)


challenge_set.to_csv("challenge_set_addParq_addApt.csv")
final_submission_set.to_csv("final_submission_set_addParq_addApt.csv")