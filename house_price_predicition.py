import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Title
st.title("🏠 House Price Prediction APP")

# Load dataset
df = pd.read_csv('house_price_dataset.csv')
X = df[['Area', 'Bedroom', 'Bathroom', 'Floors']]
y = df['Price']

# Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.success("✅ Model Trained Successfully")

# Sidebar for inputs
st.sidebar.header("📥 Input Features")
Area = st.sidebar.number_input("Area (in sqft)", min_value=100, max_value=100000, value=1500)
Bedroom = st.sidebar.number_input("Number of Bedrooms", min_value=0, max_value=100, value=3)
Bathroom = st.sidebar.number_input("Number of Bathrooms", min_value=0, max_value=100, value=2)
Floors = st.sidebar.number_input("Number of Floors", min_value=0, max_value=25, value=1)

# Prediction
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame([[Area, Bedroom, Bathroom, Floors]], 
                               columns=['Area', 'Bedroom', 'Bathroom', 'Floors']) 
    predicted_price = model.predict(input_data)[0]
    buffer = predicted_price * 0.1  # 10% price range
    
    st.subheader("💰 Predicted House Price:")
    st.success(f"Estimated Price: ₹ {predicted_price:,.2f}")
    st.info(f"Expected Range: ₹ {predicted_price-buffer:,.2f} - ₹ {predicted_price+buffer:,.2f}")

    # Visualization
    st.subheader("📊 Linear Regression Visualization")
    slope = model.coef_[0]
    intercept = model.intercept_

    avg_bedroom = df['Bedroom'].mean()
    avg_bathroom = df['Bathroom'].mean()
    avg_floors = df['Floors'].mean()

    area_vals = pd.Series(range(100, 4000, 100))
    price_vals = slope * area_vals + intercept
    price_vals += (avg_bedroom * model.coef_[1] +
                   avg_bathroom * model.coef_[2] +
                   avg_floors * model.coef_[3])

    fig, ax = plt.subplots()
    ax.scatter(df['Area'], df['Price'], color='blue', label='Actual Data')
    ax.plot(area_vals, price_vals, color='red', label='Regression Line')
    ax.scatter(Area, predicted_price, color='green', label='Predicted Point', s=100)
    ax.set_xlabel('Area (sqft)')
    ax.set_ylabel('Price (₹)')
    ax.set_title('Linear Regression: Area vs Price')
    ax.legend()
    st.pyplot(fig)

# Evaluation Metrics
st.subheader("📈 Model Evaluation")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
st.write(f"MAE: ₹ {mean_absolute_error(y_test, y_pred):,.2f}")
st.write(f"MSE: ₹ {mean_squared_error(y_test, y_pred):,.2f}")

# Feature Importance
st.subheader("📊 Feature Importance")
feature_imp = pd.Series(model.coef_, index=X.columns)
st.bar_chart(feature_imp)


# About the model
st.subheader("ℹ️ About this App")
st.write("""
    This app uses a **Multiple Linear Regression** model to predict the price of a house 
    based on four features: Area, Bedrooms, Bathrooms, and Floors. 
    
    The model is trained on a dataset of synthetic house prices. 
    You can input values manually.
    
    Here:
    R² Score indicates how well the model explains the variance in the data.
    MAE is the mean  average error in predictions.
    MSE is the mean squared error of the predictions. 
    """)


