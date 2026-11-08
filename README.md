# 💰 Salary Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning regression techniques to predict employee salaries based on demographic, educational, and professional experience information.

It explores the relationship between employee characteristics and salary, implementing **Linear Regression** and **Polynomial Regression** to model salary patterns.

The project also includes data preprocessing, categorical encoding, missing-value handling, regression visualization, residual analysis, and salary prediction for new employee records.

## 🎯 Project Objectives

- Analyze employee salary data.
- Explore relationships between experience, education, and salary.
- Handle missing values in the dataset.
- Convert categorical variables into numerical representations.
- Build a Linear Regression baseline model.
- Apply Polynomial Features to capture nonlinear relationships.
- Evaluate model performance using regression metrics.
- Visualize actual versus predicted salaries.
- Predict salaries for new employee records.

## 📂 Dataset

**Dataset:** `salry.csv`

The dataset contains employee-related information used to estimate salary.

### Input Features

| Feature | Description |
|---|---|
| Age | Employee's age |
| Gender | Employee's gender |
| Education Level | Educational qualification |
| Job Title | Employee's professional designation |
| Years of Experience | Total professional experience |

### Target Variable

`Salary` – The employee's salary, which the regression models aim to predict.

## 🛠️ Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly Express
- Scikit-learn

## 🔍 Data Preprocessing

### 1. Categorical Encoding

`OrdinalEncoder` is used to convert categorical features into numerical values.

Encoded columns include:

- Gender
- Education Level
- Job Title

### 2. Missing Value Handling

Missing values are filled using the median of their respective columns.

Features processed include:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience
- Salary

### 3. Feature and Target Separation

- Input features (`X`): All columns except `Salary`.
- Target (`y`): `Salary`.

### 4. Train-Test Split

The dataset is divided into:

- Training data: 80%
- Testing data: 20%
- Random state: 42

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression is implemented as the initial salary prediction model.

It learns the relationship between employee features and salary using a linear equation:

\[
\hat{y} = \beta_0 + \beta_1x_1 + \cdots + \beta_nx_n
\]

Where:
- \(\hat{y}\) represents predicted salary.
- \(\beta_0\) represents the intercept.
- \(\beta_i\) represents the learned coefficients.
- \(x_i\) represents input features.

### 2. Polynomial Regression

Polynomial Regression is implemented using Scikit-learn's `PolynomialFeatures`.

**Model Configuration:**
- Polynomial degree: 5
- Base estimator: Linear Regression

Polynomial feature transformation expands the original input features into higher-degree combinations, allowing the regression model to capture more complex relationships.

The transformed training and testing datasets are used for model fitting and prediction.

## 📊 Model Evaluation

The project uses the following regression metrics:

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted salaries.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the same unit as salary.
- **R² Score:** Measures how well the model explains variation in salary.

The code calculates MSE for the initial Linear Regression model and RMSE and R² for the Polynomial Regression model.

## 📈 Data Visualization

The project includes several visualizations to analyze model performance and salary relationships.

### Visualizations Included

1. **Salary Prediction Scatter Plot:** Compares actual and predicted salary values.
2. **Experience vs Salary Regression Plot:** Shows the relationship between years of experience and salary.
3. **Actual vs Predicted Salary:** Visualizes Polynomial Regression predictions against actual values.
4. **Residual Plot:** Examines prediction errors.
5. **Perfect Prediction Line:** Provides a reference for comparing actual and predicted salary values.

## 🔮 Salary Prediction on New Data

The trained Polynomial Regression model is used to predict salary for new employee records.

Example:

```python
new_data = np.array([[30, 1, 4, 105, 5]])

new_data_poly = poly.transform(new_data)

predicted_salary = lr_model.predict(new_data_poly)

print(f"Predicted Salary: {predicted_salary[0]}")
```

The input values represent the employee features in the same order and encoding used during model training.

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/salary-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd salary-prediction
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### 4. Run the Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells to preprocess the data, train the regression models, evaluate performance, and predict employee salaries.

## 📁 Project Structure

```text
Salary-Prediction/
│
├── salry.csv
├── Salary_Prediction.ipynb
├── README.md
│
└── requirements.txt
```

## 🔧 Future Improvements

- Use separate training and testing evaluation consistently.
- Prevent data leakage by fitting encoders and preprocessing steps only on training data.
- Use a Scikit-learn Pipeline for preprocessing and regression.
- Compare Linear, Ridge, Lasso, and Polynomial Regression.
- Apply cross-validation and hyperparameter tuning.
- Explore different polynomial degrees to reduce overfitting.
- Add MAE, RMSE, and R² comparisons on the test dataset.
- Develop an interactive salary prediction application using Streamlit.

## 👨‍💻 Author

**Muhammad Idrees**  
AI & Machine Learning Developer

Interested in Machine Learning, Deep Learning, Computer Vision, NLP, and AI-powered applications.

⭐ If you find this project useful, consider giving the repository a star!
