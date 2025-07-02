import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit App UI
st.title("🌸 Iris Flower Species Predictor")
st.write("This app predicts the species of an Iris flower based on user input.")

# Sidebar for input features
st.sidebar.header("Input Features")
inputs = {feat: st.sidebar.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
          for feat in iris.feature_names}
input_array = np.array([list(inputs.values())])

# Display user input
st.subheader("🔍 Your Inputs")
st.write(pd.DataFrame(input_array, columns=iris.feature_names))

# Predict
probs = model.predict_proba(input_array)[0]
predicted_class = iris.target_names[np.argmax(probs)]
st.subheader("🌼 Prediction")
st.success(f"**Predicted Species:** {predicted_class.capitalize()} with {probs.max()*100:.2f}% confidence")

# Confidence bar chart
st.subheader("📊 Prediction Confidence")
prob_df = pd.DataFrame({"Species": iris.target_names, "Probability": probs})
fig1, ax1 = plt.subplots()
sns.barplot(data=prob_df, x="Species", y="Probability", palette="viridis", ax=ax1)
ax1.set_ylim(0, 1)
st.pyplot(fig1)

# Scatter plot for placement
st.subheader("📌 Feature Placement")
selected_feature = st.selectbox("Feature to compare", iris.feature_names, index=2)
next_feature = iris.feature_names[(iris.feature_names.index(selected_feature)+1)%4]
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x=selected_feature, y=next_feature, hue="species", palette="deep", alpha=0.6, ax=ax2)
ax2.scatter(inputs[selected_feature], inputs[next_feature], color="black", s=100, label="Your Input")
ax2.legend()
st.pyplot(fig2)

