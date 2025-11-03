House Price Prediction - Linear Regression from Scratch (NumPy)

This project demonstrates the complete mathematical and programmatic implementation of Multiple Linear Regression from scratch using only NumPy.

It predicts house prices based on numerical features such as square footage, number of bedrooms, bathrooms, and year built.

No machine learning libraries like scikit-learn are used - the model, gradient descent, and cost function are fully written manually.

📘 Objective
Implement Multiple Linear Regression from scratch:

Hypothesis:
    h_wb(x) = Xw + b

Cost Function (MSE):
    J(w,b) = (1 / 2m) * Σ(ŷ - y)²

Gradient Descent Updates:
    w := w - α * (1/m) * Xᵀ(Xw + b - y)
    b := b - α * (1/m) * Σ(Xw + b - y)

Feature Scaling (Z-score Normalization)
    z = (x - μ) / σ

 
🧠 Mathematical Concepts Implemented
$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (h_{w,b}(x^{(i)}) - y^{(i)})^2
$$

	
 
🧾 Dataset
Contains 13 columns:
price, bedrooms, bathrooms, sqft_living, sqft_lot,
floors, waterfront, view, condition, sqft_above,
sqft_basement, yr_built, yr_renovated
The target variable is price.

⚙️ Implementation Steps
1️⃣ Data Preprocessing
Dropped non-numeric columns (date)
Applied Z-score and mean normalization to all input features

2️⃣ Model Initialization
w = np.zeros((12,1))
b = 0

3️⃣ Cost Function
J = (1/(2*m)) * np.sum((np.dot(x,w) + b - y)**2)

4️⃣ Gradient Descent
dj_dw = (1/m) * np.dot(x.T, (np.dot(x,w) + b - y))
dj_db = (1/m) * np.sum(np.dot(x,w) + b - y)
w = w - alpha * dj_dw
b = b - alpha * dj_db

5️⃣ Training
Ran gradient descent for 1000 iterations
Stored cost values for convergence analysis

6️⃣ Prediction
y_pred = np.dot(x_pred, w) + b
📈 Results
Metric	Value
Final Cost (J)	1.3747e+11
RMSE	524,356.55
Predicted Price Example	740,246
Actual Price Example	549,900
Interpretation:
The RMSE indicates an average deviation of about $524k per prediction - reasonable for a basic linear model without regularization or non-linear terms.

🧩 Future Work
Implement Ridge and Lasso Regression (regularization)
Add polynomial and interaction terms to capture non-linear trends
Compare with Scikit-learn's LinearRegression results
Visualize:
Cost vs Iterations
Predicted vs Actual prices

🧠 Key Learnings
Hands-on understanding of:
Matrix-vector operations in regression
The mechanics of gradient descent
Vectorization to replace Python loops
Cost minimization and convergence behavior
Reinforced mathematical intuition behind supervised learning.


📂 Project Structure
📦 House-Price-Prediction
 ┣ 📜 house_price_regression.ipynb
 ┣ 📜 data.csv
 ┣ 📜 README.md





## House Price Prediction Using scikit-learn


After implementing linear regression from scratch, we also applied **Linear Regression using scikit-learn** to leverage built-in tools for preprocessing, scaling, and modeling.


### Workflow:

1. **Data Preprocessing & Scaling**
   - Cleaned the dataset (handled missing values, encoded categorical features).
   - Scaled numerical features using `StandardScaler`.


2. **Train/Test Split**
   - Split data into training and testing sets to evaluate model performance.


3. **Linear Regression**
   - Used `LinearRegression()` from `sklearn.linear_model`.
   - Fit the model on scaled training features.
   - Predicted house prices on the test set.


4. **Polynomial Regression**
   - Expanded features using `PolynomialFeatures(degree=2)` to capture non-linear relationships.
   - Fit linear regression on the polynomial features.
   

### Results:
| Model | MSE | R² |
|-------|-----------------|--------|
| Linear Regression | 271,045,851,735 | 0.204 |
| Polynomial Regression (degree=2) | 68,549,878,500 | 0.346 |


### Insights Gained:
- Using scikit-learn simplifies preprocessing, scaling, and model fitting.
- Polynomial features help capture non-linear patterns, significantly improving R².
- Scikit-learn’s pipeline enables cleaner, more maintainable workflows compared to implementing everything from scratch.

