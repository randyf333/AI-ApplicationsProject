# Predicting Car Accident Severity
AI project for Hanyang ITE3051

Gavin Pryor, Department of Financial Management, gavin.anaiah@gmail.com <br>
Randy Fu, Department of Computer Science, randyfu333@hanyang.ac.kr <br>
Fatima Zahra El Bajdali,Department of Computer Science, bfz2005@hanyang.ac.kr <br>
Marwa Errahmani Department of Computer Science, marwaerrahmani111@gmail.com <br>

Our project intends to investigate the factors in a car accident and predict the severity of the accident using real world data and deep learning techniques. Specifically, we want to highlight which factors have the greatest impact on severity and have an AI model predict how severe an accident is based on those factors. This can later be integrated into a type of alert system, where alerts can be sent to would-be drivers warning them of the road conditions and how severe an accident could be if they get into one. We intend to train a neural network model capable of classifying accident severity into multiple levels (minor, moderate, or severe).  We will use a dataset from Kaggle, which contains detailed records of traffic incidents across the United States. The dataset includes factors such as weather conditions, temperature, visibility, time of day, and road type all of which may influence accident outcomes.  

# Introduction

Road traffic accidents remain a major public-safety concern worldwide. Beyond the human cost, they create economic losses, traffic congestion, and significant pressure on emergency-response systems. Importantly, not all accidents have the same impact: some are minor incidents, while others result in serious injury or even fatalities. Being able to predict the likely severity of an accident using contextual information, such as weather, time of day, road type, visibility conditions, and local accident density, can support faster response prioritization, smarter warning systems, and more informed planning decisions for both drivers and infrastructure managers.

This project aims to develop and evaluate a machine-learning pipeline capable of predicting accident severity (minor, moderate, or severe) from real-world traffic incident data by learning patterns from environmental conditions, roadway characteristics, spatial-temporal context, and surrounding accident history. The system is designed not only to classify the severity level but also to understand how these diverse factors interact, allowing it to generalize to new situations and provide meaningful insights about the conditions under which severe accidents are most likely to occur. We also create neighborhood-level descriptors such as kernel-density estimates (KDE) and grid-cell accident statistics to capture the influence of local accident patterns. These features are then used to train deep learning classifiers alongside baseline models, allowing us to compare performance across architectures.

An additional objective of this project is to understand which factors contribute most to accident severity. After training, we conduct interpretability analyses, such as feature-importance evaluations and model-behavior visualizations, to identify the signals that most strongly shape the model’s predictions. Finally, the project includes a short demonstration video and a clear, structured technical blog that explain the model, methods, and findings, as well as a simple concept for how such a predictive system could be integrated into real-world notifications to drivers, predictive rerouting for GPS systems, or dynamic traffic-management dashboards for city planners.

# Dataset
The dataset we are using for this project is the US Accidents (2016 - 2023) dataset found on Kaggle. This dataset contains around 7.7 million accident records across 49 US states from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks.

Our target feature and what we are predicting is Severity, which is a categorical variable with possible values of 1 (least severe), 2, 3, or 4 (most severe). In this case a "severe" accident is one that causes a longer traffic delay, with heavy congestion and long backups.

Following our processing the data, we use four broad groups of features to predict severity:
1. Time and location features that capture when and where the accident occurred. Examples include time of the accident, geographic coordinates of the crash location, and a spatial grid cells representing the area surrounding the crash.

2. Weather and environmental features, which describe the driving conditions at the time of the incident. Examples are:
+ Temperature, humidity, and air pressure
+ Visibility and wind conditions
+ Rain, snow, or other forms of precipitation
+ Indicators for clean weather, windy weather, thunderstorms, or weather intensity

3. Roadway and infrastructure features, representing characteristics of the road and nearby traffic elements, such as the presence of a crossing, junction, bump, or roundabout or, whether there is a stop sign or traffic signal

4. Spatial density and contextual features, summarizing patterns of nearby accidents and the severity of past incidents in the same area, such as local accident density within small spatial grids

These groups together give the model information about time, weather, road design, and spatial context that help it predict how severe each accident is likely to be. The original dataset, along with the full list of original features and their descriptions, can be found [here](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

# Methodology
### Data Processing
To process the data, we had to drop unnecessary columns, including leaky variables and redundant time or location fields, so we could focus on the timestamp and our feature-engineered spatial variables derived from latitude and longitude. Using the raw lat/lng values, we projected coordinates into a metric CRS, created point geometries, generated multi-scale grid cells, and computed KDE-based density features to capture both local and regional accident patterns. We also handled missing values in weather and environmental columns, removed invalid or extreme outliers, and standardized continuous fields where needed. For the weather-related variables, we condensed dozens of noisy text categories into cleaner groups like weather type, intensity, and flags for windy or thunder conditions, which made the variables much easier for the model to learn from. All of this gave us a cleaner, more meaningful version of the dataset that better reflects real accident conditions and sets the groundwork for stronger modeling.

### Modeling
To predict Severity, we tested 3 different models and compared performance for each: **Random Forest**, **XGBoost**, and an **attention based neural network**. Our goal was to find the highest performing model while still having feature performance as an output for analysis.

# Evaluation & Analysis: 
Notes
+ evaluation metrics of final model and real-world interpretation metrics (what do they tell us?)
+ feature importance comparison of 3 initial models (maybe move the comparison ones up to other section)
+ feature importance of final model
+ runtime comparison of 3 initial models
+ runtime of final model
+ decision threshold discussion and real-world interpretation

Performance Summary: 

| Model          | Accuracy |
|----------------|----------|
| XGBoost        | 0.71     |
| Attention NN   | 0.6124   |
| Random Forest  | 0.555    |

1. XGBoost Results:

   
2. Attention Neural Network Results:
      Overall Metrics:

            Accuracy: 0.6124
            
            Macro Precision: 0.6015
            
            Macro Recall: 0.6401
            
            Macro F1: 0.6056


3. Random Forest Results: 




Looking at the initial results of our three models, **XGBoost** had the highest accuracy at 0.71 compared to **Random Forest** with 0.55508 and **Attention Based Neural Network** with 0.6124. These results were not unexpected, as XGBoost excels at structued and tabular data compared to Random Forest and an Attention Based Neural Network. Given these results, we decided to focus on improving our XGBoost model's performance as described in the modeling section. To do so, we created a parameter sweep using Wandb for analysis to try and find the optimal parameters for our XGBoost model given our dataset. However, after running through the sweep the accuracy of the model did not dramatically increase, instead hovering around 0.72. The results of our sweeps are shown in the following graphs:

![f1-weighted graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154023.png)
![f1-macro graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154119.png)
![accuracy graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154137.png)
![raw f1-score graph](https://github.com/randyf333/AI-ApplicationsProject/blob/main/visualizations/Screenshot%202025-12-09%20154058.png)

This tells us that our current features may not carry enough predictive power to increase our prediction accuracy. 


# Related Work
We referenced the following sources:

- Datasets & Documentation:

    US Accidents (2016–2023) — Kaggle
  
- Methods & Technical Resources:

    XGBoost official documentation
    
    Scikit-learn documentation
    
    TensorFlow/Keras guides for custom layers
    
    Blog posts/tutorials on:
    
    Attention mechanisms for tabular data

- Tools Utilized

    Python 3.10+
    
    Pandas, NumPy, Matplotlib, Seaborn
    
    scikit-learn
    
    TensorFlow / Keras
    
    XGBoost
    
    Weights & Biases 
    
    PyTorch

# Conclusion
Notes
+ summary and wrap up
+ real world impact and applications
+ deployment feasability

Through this project, we were able to explore how different environmental, temporal, and spatial factors contribute to the severity of car accidents. By testing multiple machine learning models, we gained a better understanding of what types of algorithms work best for this kind of structured, real world data.

Overall, XGBoost performed the best, reaching about 71% accuracy, which makes sense because boosted tree models are known to handle tabular and mixed type features very well. Our attention based neural network did not reach the same level of accuracy, but it was still useful because it helped us interpret which features the model considered important. The Random Forest served as a good baseline but showed lower performance compared to the other two.

From the results, we found that weather conditions, visibility, and local accident density played a big role in predicting severity. Features related to road infrastructure and time of day also contributed, but to a smaller degree. One of the challenges we faced was the moderate severity class, which was harder to predict correctly due to overlapping patterns with the other classes.


