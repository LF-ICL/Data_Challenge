import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
from datetime import date, timedelta
import os

def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)

def extractParquet(df,fname):
    parquet_temp_df = pd.read_parquet('FlightData/2022-04-01.parquet')
    alt = 10000
    vr = 300
    df_ICA_agg = pd.DataFrame(columns = parquet_temp_df.columns)
    df_maxAlt_agg = pd.DataFrame(columns = parquet_temp_df.columns)
    #loop over all flights
    # print(df.head(5))
    #pull flight_id from challenge_set

    start_date = date(2022,10,23)
    # end_date   = date(2022,10,24)
    end_date   = date(2023, 1, 1)

    total_failed_count  = 0
    total_entries_count = 0
    datelist = []
    for single_date in daterange(start_date, end_date):
        date_find = single_date.strftime("%Y-%m-%d") 
        df_ICA = pd.DataFrame(columns = df_ICA_agg.columns)
        df_maxAlt = pd.DataFrame(columns = df_ICA_agg.columns)
        if not(os.path.isfile('FlightData/'+date_find+'.parquet')):
            datelist.append(date_find)
        else:
            parquet_df = pd.read_parquet('FlightData/'+date_find+'.parquet')
            find_df = df.loc[df['date'] == date_find,['flight_id','date']]

            failed_count = 0
            # for i in range(2):
            for i in range(len(find_df)):
                extract_df = parquet_df.loc[(parquet_df['flight_id'] == find_df.iloc[i,0]) & (parquet_df['altitude'] > alt) & (parquet_df['vertical_rate'] < vr),:]
                
                #write
                if len(extract_df)==0:
                    df_ICA.loc[len(df_ICA)] = pd.Series(dtype='float64')
                    df_ICA.loc[len(df_ICA)-1,'flight_id'] = find_df.iloc[i,0]
                    df_maxAlt.loc[len(df_maxAlt)] = pd.Series(dtype='float64')
                    df_maxAlt.loc[len(df_maxAlt)-1,'flight_id'] = find_df.iloc[i,0]
                    print(i,"/",len(find_df),": The flight_id ", find_df.iloc[i,0]," was empty")
                    failed_count = failed_count + 1
                else:
                    # print(i,"/",len(find_df),": Successful pull. flight_id ", find_df.iloc[i,0])
                    df_ICA.loc[len(df_ICA)] = extract_df.iloc[0, :]
                    # print(extract_df["altitude"].max())
                    # print(extract_df.loc[extract_df["altitude"]==extract_df["altitude"].max(), :])
                    extMaxAlt_df = extract_df.loc[extract_df["altitude"]==extract_df["altitude"].max(), :]
                    df_maxAlt.loc[len(df_maxAlt)] = extMaxAlt_df.iloc[0, :]
                    # df_maxAlt.loc[len(df_maxAlt)] = extract_df.iloc[extract_df["altitude"]==extract_df["altitude"].max(), :]
                    
            total_failed_count  = total_failed_count  + failed_count
            total_entries_count = total_entries_count + len(find_df)
            print(date_find,": ",failed_count,"/",len(find_df)," was empty")
            # print: ICA, max alt
            df_ICA_agg = pd.concat([df_ICA_agg,df_ICA]) 
            df_maxAlt_agg = pd.concat([df_maxAlt_agg,df_maxAlt])
            df_ICA_agg.to_csv(fname+"_ICAs.csv")
            df_maxAlt_agg.to_csv(fname+"_maxAlts.csv")
    print("total failed count: ",total_failed_count,"/",total_entries_count)
    print("missing dates:", datelist)

    return datelist, total_failed_count, total_entries_count

def __main__():    
    df = pd.read_csv('challenge_set.csv')
    chaMissingdates, chaFailed, chaEntries = extractParquet(df,"challenge_set")
    df = pd.read_csv('final_submission_set.csv')
    finMissingdates, finFailed, finEntries = extractParquet(df,"final_submission_set")

    print("Challenge Set:")
    print("total failed count: ",chaFailed,"/",chaEntries)
    print("missing dates:", chaMissingdates)

    print("Final Submission Set:")
    print("total failed count: ",finFailed,"/",finEntries)
    print("missing dates:", finMissingdates)

__main__()
