from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Created by Team_Jolly_Koala

def assignAverageToOneCat(df,cat,param):
    mean_group_by_cat = df[[cat,param]].groupby([cat])
    mean_group_by_cat = mean_group_by_cat.mean()
    cat_param = mean_group_by_cat[param].values
    cat_list = mean_group_by_cat[param].index.values    
    cat_dic = {}
    for i in range(len(cat_list)):
        cat_dic[cat_list[i]] = cat_param[i]
    return cat_dic

def cleanParamNaN(df,param,newVal):    
    print(df[param].isna().sum(), "rows of NaN from "+df.name+" is filled with "+param+"="+str(newVal))
    # df.loc[df[param].isna(), [param]] = newVal
    df[param] = df[param].fillna(newVal)

# ---------- ---------- ---------- ------- ----------
# ---------- Inputs: Design of Experiments ----------
featureParamSet1 = ["actype_meantow","acOp_meantow","flight_duration","flown_distance","taxiout_time","ICA","maxAlt","month","ades_lon","ades_lat","adep_lon","adep_lat"]
featureParamSet2 = "des_dev"
resultParam = "tow"
# n_estimators...
# The randomness of the first ExtraTree - so that the first tree is not overfitting and reflects the norm best to prepare for the representativeness of des_dev
# Any training models can give priority to different params?
# ---------- Inputs: Design of Experiments ----------
# ---------- ---------- ---------- ------- ----------

# Load input data
challenge_set =pd.read_csv("challenge_set_addParq_addApt.csv")
challenge_set['month']=pd.DatetimeIndex(challenge_set['date']).month
challenge_set['acOp'] = challenge_set[['aircraft_type', 'airline']].apply(tuple, axis=1)
challenge_set.name = "challenge_set"
final_submission_set =pd.read_csv("final_submission_set_addParq_addApt.csv")
final_submission_set['month']=pd.DatetimeIndex(final_submission_set['date']).month
final_submission_set['acOp'] = final_submission_set[['aircraft_type', 'airline']].apply(tuple, axis=1)
final_submission_set.name = "final_submission_set"

# ---------------Version 16 (Section 1)--------------- 
# V16: Branch from V12 but include more features: "month","ades_lon","ades_lat","adep_lon","adep_lat"
# V12: airline/aircraft type categorical
acOp_tow_dic = assignAverageToOneCat(challenge_set,"acOp","tow")
challenge_set["acOp_meantow"] = challenge_set["acOp"].map(acOp_tow_dic)
final_submission_set["acOp_meantow"] = final_submission_set["acOp"].map(acOp_tow_dic)
actype_tow_dic = assignAverageToOneCat(challenge_set,"aircraft_type","tow")
challenge_set["actype_meantow"] = challenge_set["aircraft_type"].map(actype_tow_dic)
final_submission_set["actype_meantow"] = final_submission_set["aircraft_type"].map(actype_tow_dic)

cleanParamNaN(challenge_set,"ICA",35000)
cleanParamNaN(challenge_set,"maxAlt",38000)
cleanParamNaN(challenge_set,"ades_lon",0)
cleanParamNaN(challenge_set,"ades_lat",0)
cleanParamNaN(challenge_set,"adep_lon",0)
cleanParamNaN(challenge_set,"adep_lat",0)
cleanParamNaN(final_submission_set,"ICA",35000)
cleanParamNaN(final_submission_set,"maxAlt",38000)
cleanParamNaN(final_submission_set,"acOp_meantow",0)
cleanParamNaN(final_submission_set,"ades_lon",0)
cleanParamNaN(final_submission_set,"ades_lat",0)
cleanParamNaN(final_submission_set,"adep_lon",0)
cleanParamNaN(final_submission_set,"adep_lat",0)

# ---------------Version 16 (section 2) ---------------
# 1. Train ExtraTrees once to, and generate error term on tow with FeatureSet1
# V16: Include ICA and maxAlt effects
# trainParam = featureParamSet1
# trainParam.append(resultParam)
# train_df = challenge_set[trainParam]
X = challenge_set[featureParamSet1].to_numpy()
y = challenge_set[resultParam].to_numpy()
print("Training Tree 1...")
ExTreeReg1_v16 = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X, y)

# Feed back on residual
print("Generating Error Term from Tree 1")
training_set_pred = ExTreeReg1_v16.predict(X)
challenge_set["tow_pred"] = training_set_pred
challenge_set["tow_diff"] = challenge_set["tow_pred"].sub(challenge_set["tow"], axis=0)

# [Optional] Quality Check of 1st Extra Trees
print("Cross Val-ing Tree 1...")
score = cross_val_score(ExTreeReg1_v16, X, y, scoring='neg_root_mean_squared_error').mean().round(2)
print("ExTreeReg1_v16 cross val score is: ",score)
# cross val score is:  -4095.77 (v12) --> -3830.65 (v16)

# [Optional] Interim Output of 1st Extra Trees
X = final_submission_set[featureParamSet1]
print("Generating Final Submission Prediction from Tree 1")
final_submission_set_pred = ExTreeReg1_v16.predict(X)
final_submission_set["tow"] = final_submission_set_pred
final_submission_set[["flight_id","tow"]].to_csv("final_submission_set_ExTreeReg1_v16.csv",index=False)
### RMSE score on OSN Ranking = 

# ---------------Version 16 (Section 3) ---------------
# 2. Create des_dev (residual averaged against destination)
# characterise destination airport by the residuals from previous best training.
ades_dev_dic = assignAverageToOneCat(challenge_set,"ades","tow_diff")
challenge_set["des_dev"] = challenge_set["ades"].map(ades_dev_dic)
final_submission_set["des_dev"] = final_submission_set["ades"].map(ades_dev_dic)

cleanParamNaN(challenge_set,"des_dev",0)
cleanParamNaN(final_submission_set,"des_dev",0)

# ---------------Version 16 (section 4) ---------------
# 3. Run ExtraTrees again to predict answer with FeatureSet2 # (including des_dev and those 
#       from FeatureSet1, but perhaps other params that were not included in FeaturesSet1)
featureParamSet1.append(featureParamSet2)
X = challenge_set[featureParamSet1].to_numpy()
y = challenge_set[resultParam].to_numpy()
print("Training Tree 2...")
ExTreeReg2_v16 = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X, y)

# [Optional] Quality Check of 2nd Extra Trees
print("Cross Val-ing Tree 2...")
score = cross_val_score(ExTreeReg2_v16, X, y, scoring='neg_root_mean_squared_error').mean().round(2)
print("ExTreeReg2_v16 cross val score is: ",score)
# cross val score is:  -4095.77

# Final Output of 2nd Extra Trees
X = final_submission_set[featureParamSet1]
print("Generating Final Submission Prediction from Tree 2")
final_submission_set_pred = ExTreeReg2_v16.predict(X)
final_submission_set["tow"] = final_submission_set_pred
final_submission_set[["flight_id","tow"]].to_csv("final_submission_set_ExTreeReg2_v16.csv",index=False)
### RMSE score on OSN Ranking = 