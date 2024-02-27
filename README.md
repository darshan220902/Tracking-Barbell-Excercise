## Tracking Barbell Excercises with Sensor Data



🔍 **Dataset Information:**

* Data collected from 5 individuals.
* Captured accelerometer and gyroscope readings over 3 axes (x, y, z).
* Exercise labels and participant names extracted from file names.



🛠️ **Key Steps:**

* Import & Organization: Loaded sensor data, organized it, and extracted participant and exercise information.

* Cleaning & Resampling: Removed unnecessary columns, resampled data, and addressed missing values.

* Outlier Detection: Utilized IQR and Chauvenet's Criterion to identify outliers.

* Feature Engineering: Built various feature sets including basic, square, PCA, temporal, frequency, and clustering features.

* Model Evaluation: Explored the performance of classification models such as Neural Network, Random Forest, K-Nearest Neighbors, Decision Tree, and Naive Bayes.

* Results & Insights: Found Random Forest to be the most accurate model, achieving an accuracy of 99%.

* Streamlit: Incorporate Streamlit for interactive visualization, enabling seamless exploration of the project's insights and results.



🎯 **Project Purpose:**

* The objective was to develop a robust system for tracking barbell exercises, enabling better understanding and monitoring of exercise performance.

* This system has potential applications in fitness tracking apps, personalized workout recommendations, and performance analysis tools.



🚀 **Outcome:**

* Created a model capable of accurately predicting exercise labels based on accelerometer and gyroscope data.

* This project contributes to fitness, health, and athletic training objectives by providing valuable insights into exercise performance.



