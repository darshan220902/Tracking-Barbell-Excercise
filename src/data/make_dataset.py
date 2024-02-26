import pandas as pd
from glob import glob

df=pd.read_csv('../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')
df

files=glob('../../data/raw/MetaMotion/*.csv')
len(files)


#files[0]

data_path="../../data/raw/MetaMotion\\"
##f=files[0]
##f



def read_data(files):
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()

    acc_set=1
    gyr_set=1

    data_path="../../data/raw/MetaMotion\\"
    for f in files:
        participent=f.split("-")[0].replace(data_path,"")
        label=f.split("-")[1]
        condition=f.split("-")[2].rstrip('123').rstrip('_MetaWear_2019')

        df=pd.read_csv(f)

        df['participent']=participent
        df['label']=label
        df['condition']=condition
        
        if "Accelerometer" in f:
            df['set']=acc_set
            acc_set+=1
            acc_df=pd.concat([acc_df,df])

        if "Gyroscope" in f:
            df['set']=gyr_set
            gyr_set+=1
            gyr_df=pd.concat([gyr_df,df])
    
    acc_df.index=pd.to_datetime(acc_df['epoch (ms)'],unit="ms")
    gyr_df.index=pd.to_datetime(gyr_df['epoch (ms)'],unit="ms")
    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    del gyr_df['epoch (ms)']
    del gyr_df['time (01:00)']
    del gyr_df['elapsed (s)']

    return acc_df,gyr_df

acc_df,gyr_df=read_data(files)
#acc_df[acc_df['set']==92]

pd.concat([acc_df,gyr_df],axis=1) # duplicates name of column and higher null values
data_merge=pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1) # take only feasible columns
data_merge.dropna() # only 1119 rows are left from 69677 so there is major lost of data we don't directly drop null

data_merge.columns=[
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participent",
    "label",
    "condition",
    "set"
]
sampling={
    "acc_x":'mean',
    "acc_y":'mean',
    "acc_z":'mean',
    "gyr_x":'mean',
    "gyr_y":'mean',
    "gyr_z":'mean',
    "participent":'last',
    "label":'last',
    "condition":'last',
    "set":'last'
}


##--- Sampling 
data_merge.info()

##data_merge[:].resample(rule='200ms').apply(sampling)


days=[g for n,g in data_merge.groupby(pd.Grouper(freq='D'))]
days[0]
data_sampler=pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])


data_sampler

data_sampler['set']=data_sampler['set'].astype('int')


data_sampler.info()

data_sampler.head()

data_sampler.to_pickle("../../data/interim/01_processed_data.pkl")
