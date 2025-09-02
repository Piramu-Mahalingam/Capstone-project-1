import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# -------------------------------
# Load Dataset
# -------------------------------
  # <--- replace with your filename if different

def load_data():
    df = pd.read_csv("House Price.csv")   # <--- replace with your filename if different
    return df

df = load_data()


st.title("📊 Student Segmentation & Profiling (Clustering App)")
st.write("This app performs **EDA, preprocessing, clustering (KMeans, Hierarchical, DBSCAN)**, and compares results.")

# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------
st.header("1️⃣ Exploratory Data Analysis")
st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Basic Info")
st.write(df.describe())

# Plot distributions
num_cols = df.select_dtypes(include=np.number).columns.tolist()
st.subheader("Distribution of Numerical Features")
for col in num_cols[:5]:   # show first 5 for speed
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# Skewness + Transformation
# -------------------------------
st.header("2️⃣ Skewness & Transformation")
skewness = df[num_cols].skew()
st.write(skewness)

# Apply log1p to positively skewed columns
skewed_cols = [c for c in num_cols if skewness[c] > 0.75 and df[c].min() >= 0]
df_trans = df.copy()
for c in skewed_cols:
    df_trans[c] = np.log1p(df[c])
st.write(f"Applied log transformation to: {skewed_cols}")

# -------------------------------
# Outlier Treatment + Scaling
# -------------------------------
st.header("3️⃣ Outlier Treatment & Scaling")

def iqr_cap(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return series.clip(lower, upper)

for c in num_cols:
    df_trans[c] = iqr_cap(df_trans[c])

scaler = StandardScaler()
X = scaler.fit_transform(df_trans[num_cols].fillna(0))

st.write("✅ Outliers capped & features scaled.")

# -------------------------------
# Clustering Models
# -------------------------------
st.header("4️⃣ Clustering Models & Evaluation")

results = []

# K-Means
best_score = -1
best_k = None
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    results.append(("KMeans", k, score))
    if score > best_score:
        best_score = score
        best_k, best_labels = k, labels

# Agglomerative
agg = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg.fit_predict(X)
agg_score = silhouette_score(X, agg_labels)
results.append(("Agglomerative", best_k, agg_score))

# DBSCAN
db = DBSCAN(eps=2.5, min_samples=5)
db_labels = db.fit_predict(X)
if len(set(db_labels)) > 1:
    db_score = silhouette_score(X, db_labels)
else:
    db_score = -1
results.append(("DBSCAN", "auto", db_score))

# -------------------------------
# Results
# -------------------------------
results_df = pd.DataFrame(results, columns=["Method", "Params", "Silhouette Score"])
st.dataframe(results_df)

best_method = results_df.loc[results_df["Silhouette Score"].idxmax()]
st.success(f"🏆 Best method: {best_method['Method']} with silhouette = {best_method['Silhouette Score']:.3f}")

# Add labels to dataset
df["Cluster"] = best_labels if best_method["Method"]=="KMeans" else (
    agg_labels if best_method["Method"]=="Agglomerative" else db_labels
)

st.header("5️⃣ Clustered Data Preview")
st.write(df.head())

# Visualization of Clusters (first 2 numerical features)
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=df["Cluster"], palette="tab10", ax=ax)
st.pyplot(fig)

# Download clustered dataset
csv = df.to_csv(index=False).encode()
st.download_button("⬇️ Download Clustered Dataset", csv, "clustered_students.csv", "text/csv")
