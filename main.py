import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.features.remove_outliers import mark_outliers_chauvenet, mark_outliers_iqr, plot_binary_outliers
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)
def display_dataframes(original_path):
    # Load the original DataFrame

    original_df = pd.read_pickle(original_path)
    
    st.subheader("Dataset Overview")
    st.markdown("""
    **Data Source:**
                
       * Utilizing MetaMotion sensors for data collection.
                
       * Sensors include accelerometer and gyroscope modules.

    **Sensor Outputs:**
                
       * Accelerometer: Captures acceleration forces experienced by the sensor.  
                    
       * Gyroscope: Measures the rate of rotation around different axes.

    **Dataset  Information :**
                
    * Data collected from 5 individuals.
        * Names of participants involved in the study.
                
        * Labels assigned to each exercise (e.g., bench press, over head press, deadlift).
                
        * Categorization based on exercise intensity (e.g., heavy, medium).
                
        * Data recorded over 3 axes (x, y, z) for both accelerometer and gyroscope.

    """)
    st.write(original_df)

def visualize_data(df):
    st.subheader("Visualisation")
    st.markdown(""" 
    **Combined Accelerometer & Gyroscope Plots:**
                
    1. Synthesized information by creating combined plots for both accelerometer and gyroscope data.
    2. Enabled a holistic view of sensor readings.
    """)
    participents=df["participent"].unique()
    labels=df["label"].unique()

    
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
                st.write(f"{label.title()}_{participent.title()}")
                st.pyplot(fig)
                plt.close()

def build_feature(df_build):
    st.subheader("Build Feature's")
    st.write("Build Feature DataFrame")
    st.write(df_build)
    df_cluster = df_build.copy()
    cluster_columns = ["acc_x", "acc_y", "acc_z"]
    k_values = range(2, 10)
    inertias = []

    st.write("Select the value of K")
    for k in k_values:
        subset = df_cluster[cluster_columns]
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        cluster_labels = kmeans.fit_predict(subset)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 10))
    plt.plot(k_values, inertias)
    plt.xlabel("k")
    plt.ylabel("Sum of Squared Distance")
    st.pyplot(plt)
    st.markdown('''
    **Clustering Analysis:**
                
    1. Employed KMeans clustering with varying 'k' values and visualized clusters in a 3D plot based on accelerometer and Gyroscope data.
    ''')
    st.write("Cluster Data on the Basis of Excercise Label")
    kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
    subset = df_cluster[cluster_columns]
    df_cluster["cluster"] = kmeans.fit_predict(subset)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")

    for l in df_cluster["label"].unique():
        subset = df_cluster[df_cluster["label"] == l]
        ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
    ax.set_xlabel("X_axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.legend()
    st.pyplot(fig)

def model(df,score):
    st.subheader("Model Evalution")

    st.markdown('''**Train & Test Data**''')
    df_train=df.drop(["participent","condition","set"],axis=1)

    X=df_train.drop(["label","index"],axis=1)
    y=df_train["label"]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

    fig, ax = plt.subplots(figsize=(10, 5))

    df_train["label"].value_counts().plot(
        kind="bar", ax=ax, color="lightblue", label="Total"
    )

    y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")

    y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")

    plt.legend()
    st.pyplot(fig)


    d=score.sort_values(by="accuracy",ascending=False)
    
    st.markdown(''' 
        **Model Comparison**:
                
        ***Explored the performance of various classification models:***
                
        1)Neural Network (NN)
                
        2)Random Forest (RF)
                
        3)K-Nearest Neighbors (KNN)
                
        4)Decision Tree (DT)
                
        5)Naive Bayes (NB)
                
        ''')
    
    st.write(d)
    
    st.markdown('''
    **Feature Set Impact:**
                
    1. Investigated the influence of different feature sets on model accuracy.
                
    2. Analyzed accuracy scores for each model across various feature sets.
    ''')

    plt.figure(figsize=(10,10))
    sns.barplot(x="model",y="accuracy",hue="feature_set",data=score)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7,1)
    plt.legend(loc="lower right")
    st.pyplot()


def main():
    original_path = "E:/Tracking_Barbell_Excercises/data/interim/01_processed_data.pkl"

    # Sidebar
    st.sidebar.title("Module")
    mode = st.sidebar.selectbox("Select Mode", ["Project Purpose","Dataset Overview", "Visualization","Outlier","Outlier Handling","Build Feature's","Model Evalution","Results and Insights"])

    # Main content based on mode selection
    if mode=="Project Purpose":
        st.markdown("## Project Purpose")

        st.markdown("""
        **Objective**: 
                    
          *  Develop a system for tracking barbell exercises using accelerometer and gyroscope data collected from MetaMotion sensors.

        **Application**:
                    
          * Once developed and validated, the model can be applied to real time or batch processing scenarios.
                    
          * It can provide valuable insights into exercise performance, form, and technique.
                    
          * Potential applications include fitness tracking apps, personalized workout recommendations, and performance analysis tools.

        **Project Outcome**:
                    
          * The project aims to create a robust and accurate system for tracking barbell exercises using sensor data.
                    
          * It facilitates better understanding and monitoring of exercise performance, contributing to fitness, health, and athletic training objectives.
                    
        """)
    elif mode == "Dataset Overview":
        display_dataframes(original_path)
    elif mode == "Visualization":
        # Load data for visualization
        df = pd.read_pickle(original_path)
        visualize_data(df)
        
    elif mode=="Outlier":
        st.subheader("Outlier's Boxplot")
        df=pd.read_pickle(original_path)
        outlier_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        st.write("Accelometer")
        fig1, ax1 = plt.subplots(figsize=(20, 10))
        df[outlier_columns[:3]].assign(label=df['label']).boxplot(by="label", ax=ax1)
        plt.title("Box Plot of First Three Outlier Columns by Label")
        plt.suptitle("")  # Suppress default title
        plt.xlabel("Label")
        plt.ylabel("Value")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

        # Display box plots in Streamlit
        st.pyplot(fig1)

        # Create box plots for remaining outlier columns
        st.write("Gyroscope")
        fig2, ax2 = plt.subplots(figsize=(20, 10))
        df[outlier_columns[3:]].assign(label=df['label']).boxplot(by="label", ax=ax2)
        plt.title("Box Plot of Remaining Outlier Columns by Label")
        plt.suptitle("")  # Suppress default title
        plt.xlabel("Label")
        plt.ylabel("Value")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        st.pyplot(fig2)
    elif mode == "Outlier Handling":
        # Load data
        df = pd.read_pickle(original_path)
        outlier_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']  # Define your outlier columns
        outlier_method = st.sidebar.radio("Select Outlier Detection Method", ["IQR", "Chauvenet"])
        st.subheader(f"Outliers's detection by {outlier_method}")
        for col in outlier_columns:
            # Call the plot_binary_outliers function
            if outlier_method == "IQR":
                dataset = mark_outliers_iqr(df, col)
                g = plot_binary_outliers(dataset, col=col, outlier_col=col+"_outlier", reset_index=True)
                st.pyplot(g)
            elif outlier_method == "Chauvenet":
                
                dataset = mark_outliers_chauvenet(df, col)
                fig = plot_binary_outliers(dataset, col=col, outlier_col=col+"_outlier", reset_index=True)
                st.pyplot(fig)
            else:
                st.warning("Please select a valid outlier detection method.")
    elif mode == "Build Feature's":
        # Load data for building features
        df = pd.read_pickle('E:\Tracking_Barbell_Excercises/data/interim/03_data_features.pkl')
        build_feature(df)
    
    elif mode == "Model Evalution":
        df = pd.read_pickle('E:\Tracking_Barbell_Excercises/data/interim/03_data_features.pkl')
        score=pd.read_pickle('E:\Tracking_Barbell_Excercises/data/interim/score_df.pkl')
        model(df,score)
    
    elif mode=="Results and Insights":
        st.markdown('''
        **Results and Insights**
                    
        1.The accuracy of RF is too good is about 0.99 and the least suitable model is KNN with accuracy of 0.78.
                    
        2.This Model is Capable the label the exercise very effectively by taking input data of participant.
        
        ''')


if __name__ == "__main__":
    main()
