import mlflow
#from src.data import make_dataset
from src.visualization import visualize
from src.models import train_model


def main():
    # Load your pickle files
    data1 = '/data/interim/01_processed_data.pkl'
    data2 = '/data/interim/02_outliers_removed_chauvenets.pkl'
    data3 = '/data/interim/03_data_features.pkl'
    # Run your data processing script for each dataset
    
    
    # Visualize your data for each dataset
    visualize.visualize_data(data1)
    
    
    # Train your model for each dataset
    #train_model.train(data1)
    #train_model.train(data2)
    #train_model.train(data3)
    
    # Log MLflow experiments
    mlflow.start_run()
    #mlflow.log_param("param_name", param_value)
    #mlflow.log_metric("metric_name", metric_value)
    mlflow.end_run()

if __name__ == "__main__":
    main()
