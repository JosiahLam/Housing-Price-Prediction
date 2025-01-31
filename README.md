# 🏡 Machine Learning Project - Housing Price Prediction  

This project explores **machine learning regression techniques** to predict housing prices based on various features such as house area, number of bedrooms, furnishing status, and proximity to main roads. It demonstrates key concepts in **data preprocessing, ordinal and nominal encoding, and hyperparameter tuning** to optimize model performance.  

## 🔍 Project Overview  

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset). The primary goal is to analyze how different regression models perform in predicting house prices while addressing challenges such as **multicollinearity** and **hyperparameter tuning**.  

### 🏰 Models Used  

Four regression models were implemented and fine-tuned to improve the **R² (R-squared) score**:  

1. **Multiple Linear Regression**  
2. **Decision Tree Regression**  
3. **Random Forest Regression**  
4. **Support Vector Machine (SVM) Regression**  

## 📊 Results & Performance  

| Model                        | R² Score  |
|------------------------------|----------|
| **Multiple Linear Regression** | 0.6781   |
| **Decision Tree Regression**  | 0.4815   |
| **Random Forest Regression**  | 0.6665   |
| **Support Vector Machine (SVM)** | 0.6482   |

- By default, an **80:20 train-test split** was used.  
- The **Multiple Linear Regression model** was also tested with a **70:30 split**, which provided additional insights into performance variations.  
- **Multicollinearity** was observed, adding complexity to the hyperparameter tuning process.  

## 🚀 Performance Improvement Strategies  

To enhance model accuracy and robustness, the following approaches can be explored:  

✔️ **Outlier Removal** – Identifying and eliminating outliers to improve model generalization.  
✔️ **Optimizing Train-Test Split Ratio** – Experimenting with different splits (e.g., 70:30, 75:25) to find the best balance between training and validation performance.  
✔️ **Overfitting vs. Underfitting Analysis** – Adjusting hyperparameters and regularization techniques to find an optimal trade-off.  

## 💂️ Dataset  

- **Source**: [Kaggle - Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)  
- **Features include**:  
  - House area  
  - Number of bedrooms  
  - Furnished status  
  - Proximity to main roads  
  - And more...  

## 🔧 Installation & Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Housing-Price-Prediction.git
   cd Housing-Price-Prediction
   ```
2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:  
   ```bash
   jupyter notebook
   ```
4. Open `Model_01.ipynb` and follow the steps for model training and evaluation.  

## 📌 Conclusion  

This project provides valuable insights into how different regression models perform in predicting housing prices. It highlights the importance of **feature selection, hyperparameter tuning, and dataset preprocessing** in achieving better model performance.  
