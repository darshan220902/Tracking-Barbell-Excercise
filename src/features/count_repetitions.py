import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle('../../data/interim/01_processed_data.pkl')
df=df[df["label"]!="rest"]

acc_r=df["acc_x"] **2 + df["acc_y"] **2 + df["acc_z"] **2

gyr_r=df["gyr_x"] **2 + df["gyr_y"] **2 + df["gyr_z"] **2

df["acc_r"]=np.sqrt(acc_r)
df["gyr_r"]=np.sqrt(gyr_r)

# -  -------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df=df[df["label"]=="bench"]
row_df=df[df["label"]=="row"]
ohp_df=df[df["label"]=="ohp"]
squat_df=df[df["label"]=="squat"]
dead_df=df[df["label"]=="dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df='h'

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------