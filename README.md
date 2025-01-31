# Machine Learning Project - Regression

This side project demonstrates machine learning concepts and data preprocessing, ordinal and nominal encoding. The objective is to predict the housing price based on certain factors like house area, bedrooms, furnished, nearness to the main road, etc. Each of the four models' parameters is hyper-tuned to optimize the R square score. The following are the models used to train the model (Model 1 to 4 respectively): 

>- Mult-Linear Regression
>- Decision Tree Regression
>- Random Forest Regression
>- Support Vector Machine (SVM)

This project is shows useful insight in terms of different variables when predicting housing prices. When hyper-tuning parameters on each model, it shows a complexity arises due to the noticeable strong multicollinearity. For default, an 80:20 split is used when splitting the dataset into training and test sets. To improve performance, the mult-linear regression model was also tested on 70:30 split. 

Dataset Link: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

R Square Score for each model 

| Model Name | R Sqaure Score |
| -------- | ------- |
| Mult-Linear Regression | 0.6781 |
| Decision Tree Regression | 0.4815 |
| Random Forest Regression | 0.6665 |
| Support Vector Machine (SVM)  | 0.6482 |

Performance Improvement: 
>- Remove Outliers
>- Explore different train and test set split ratio
>- Overfitting or Underfitting analysis 
