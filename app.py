# ✅ STEP 1: Write Streamlit app code to app.py
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data and train
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

# Sidebar sliders
st.sidebar.header("Input Flower Features")
inputs = {feat: st.sidebar.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
          for feat in iris.feature_names}
input_array = np.array([list(inputs.values())]).reshape(1, -1)

# Show input values
st.subheader("🌸 Your Input Features")
st.write(inputs)

# Prediction
probs = model.predict_proba(input_array)[0]
pred = iris.target_names[np.argmax(probs)]
st.subheader("✨ Prediction")
st.success(f"**{pred.capitalize()}** with {probs.max()*100:.2f}% confidence")

# Visualization: Probability bar chart
st.subheader("📊 Prediction Confidence")
prob_df = pd.DataFrame({"species": iris.target_names, "probability": probs})
fig1, ax1 = plt.subplots()
sns.barplot(data=prob_df, x="species", y="probability", palette="viridis", ax=ax1)
ax1.set_ylim(0, 1)
st.pyplot(fig1)

# Visualization: Where input sits in feature space
st.subheader("📌 Input Feature Placement")
selected_feature = st.selectbox("Feature to plot vs Others", iris.feature_names, index=2)
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x=selected_feature, y=iris.feature_names[(iris.feature_names.index(selected_feature)+1)%4],
                hue="species", palette="deep", ax=ax2, alpha=0.6)
ax2.scatter(inputs[selected_feature], inputs[iris.feature_names[(iris.feature_names.index(selected_feature)+1)%4]],
            c="black", s=100, label="Your Input")
ax2.legend()
st.pyplot(fig2)


# ✅ STEP 2: Launch Streamlit with Ngrok tunnel
from pyngrok import ngrok
import threading
import time

# Kill any running Streamlit processes
!pkill streamlit

# Start ngrok tunnel

public_url = ngrok.connect(8501)
print(f"🌐 Your Streamlit app is live at: {public_url}")

# Run Streamlit app in background
def run():
    os.system('streamlit run app.py')

thread = threading.Thread(target=run)
thread.start()
time.sleep(5)
