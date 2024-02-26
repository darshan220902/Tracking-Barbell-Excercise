import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mlt 
from IPython.display import display

df=pd.read_pickle("E:\Tracking_Barbell_Excercises/data/interim/01_processed_data.pkl")

df.head()

set_df=df[df["set"]==1]

plt.plot(set_df["acc_y"])
 

for l in df["label"].unique():
    subset=df[df['label']==l]
    fig,ax=plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True),label=l)
    plt.legend()
    plt.show()
   

for l in df["label"].unique():
    subset=df[df['label']==l]
    fig,ax=plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True),label=l)
    plt.legend()
    plt.show()
   
#mlt.style.use["seaborn-deep"]
mlt.rcParams["figure.figsize"]=(20,5)
mlt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"] = 3

category_df=df.query("label=='squat'").query("participent=='A'").reset_index(drop=True)
fig,ax=plt.subplots()
category_df.groupby(['condition'])["acc_y"].plot()
plt.legend()
plt.xlabel('Sample')
plt.ylabel('acc_y')


participent_df=df.query("label=='bench'").sort_values('participent').reset_index(drop=True)
fig,ax=plt.subplots()
participent_df.groupby(['participent'])["acc_y"].plot()
plt.legend()
plt.xlabel('sample')
plt.ylabel('acc_y')

participents=df["participent"].unique()
labels=df["label"].unique()

for label  in labels:
    for participent in participents:
        all_df=df.query(f"label=='{label}'").query(f"participent=='{participent}'").reset_index(drop=True)

        if len(all_df)>0:
            fig,ax=plt.subplots()
            all_df[['gyr_x','gyr_y','gyr_z']].plot(ax=ax)
            plt.legend()
            plt.xlabel('sample')
            plt.ylabel('axis')
            plt.title(f"{label} {participent}".title())


for label  in labels:
    for participent in participents:
        all_df=df.query(f"label=='{label}'").query(f"participent=='{participent}'").reset_index(drop=True)

        if len(all_df)>0:
            fig,ax=plt.subplots()
            all_df[['acc_x','acc_y','acc_z']].plot(ax=ax)
            plt.legend()
            plt.xlabel('sample')
            plt.ylabel('axis')
            plt.title(f"{label} {participent}".title())





for label  in labels:
    for participent in participents:
        combined_plot_df=df.query(f"label=='{label}'").query(f"participent=='{participent}'").reset_index(drop=True)

        if len(combined_plot_df)>0:
            fig,ax=plt.subplots(nrows=2,sharex=True,figsize=(20,10))
            combined_plot_df[['acc_x','acc_y','acc_z']].plot(ax=ax[0],linewidth=3)
            combined_plot_df[['gyr_x','gyr_y','gyr_z']].plot(ax=ax[1],linewidth=3)
            ax[0].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True)
            ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,1.150),ncol=3,fancybox=True)
            ax[1].set_xlabel('samples')
            plt.savefig(f"../../reports/figures/{label.title()} ({participent.title()}).png")
            plt.show()
