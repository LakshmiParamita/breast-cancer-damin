
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go

st.set_page_config(
    page_title="Breast Cancer Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Seeting CSS - jangan diubah
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('breast-cancer.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    try:
        data = data.drop('id', axis=1)
        
        le = LabelEncoder()
        data['diagnosis'] = le.fit_transform(data['diagnosis'])
        
        num_cols = data.select_dtypes(["float64", "int64"])
        for col in num_cols.columns:
            Q1, Q3 = data[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_range = Q1 - (1.5 * IQR)
            upper_range = Q3 + (1.5 * IQR)
            data[col] = np.where(data[col] > upper_range, upper_range, data[col])
            data[col] = np.where(data[col] < lower_range, lower_range, data[col])
        
        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def main():
    st.title('🏥 Breast Cancer Analysis Dashboard')
    
    with st.spinner('Loading data...'):
        data = load_data()
        if data is None:
            st.stop()
        processed_data = preprocess_data(data)
        if processed_data is None:
            st.stop()
    
    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select Page:', 
        ['Data Overview', 'Cancer Prediction'])
    
    if page == 'Data Overview':
        st.header('📊 Data Overview')
        
        original_data = load_data()
        diagnosis_counts = original_data['diagnosis'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Benign (B) Cases", diagnosis_counts['B'])
        with col2:
            st.metric("Malignant (M) Cases", diagnosis_counts['M'])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.pie(diagnosis_counts.values, 
                labels=['Benign', 'Malignant'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('Distribution of Diagnosis')
        st.pyplot(fig)
        
        st.subheader('Dataset Information')
        st.write(f"Jumlah sample: {len(original_data)}")
        st.write(f"Jumlah Fitur: {len(original_data.columns) - 2}")
        
        st.subheader('Feature Descriptions')
        st.markdown("""
        Penjelasan dari setiap fitur pengukuran tumor:
        
        1. **Radius**: Ukuran tumor
        2. **Texture**: Kehalusan atau kekasaran tumor
        3. **Perimeter**: Lingkar tumor
        4. **Area**: Luas permukaan tumor
        5. **Smoothness**: Variasi lokal pada panjang radius
        6. **Compactness**: Tingkat kebulatan tumor
        7. **Concavity**: Tingkat cekungan tumor
        8. **Concave points**: Jumlah lekukan pada tepi tumor
        9. **Symmetry**: Simetris tumor terhadap pusatnya
        10. **Fractal dimension**: "Tingkat detail permukaan tumor
        """)
        
    elif page == 'Cancer Prediction':
        st.header('🔍 Cancer Prediction')
        
        col1, col2 = st.columns(2)
        
        with col1:
            radius = st.number_input('Radius', min_value=0.0, max_value=50.0, value=14.0)
            texture = st.number_input('Texture', min_value=0.0, max_value=50.0, value=19.0)
            perimeter = st.number_input('Perimeter', min_value=0.0, max_value=200.0, value=92.0)
            area = st.number_input('Area', min_value=0.0, max_value=2500.0, value=654.0)
            smoothness = st.number_input('Smoothness', min_value=0.0, max_value=1.0, value=0.1)
        
        with col2:
            compactness = st.number_input('Compactness', min_value=0.0, max_value=1.0, value=0.1)
            concavity = st.number_input('Concavity', min_value=0.0, max_value=1.0, value=0.1)
            concave_points = st.number_input('Concave Points', min_value=0.0, max_value=1.0, value=0.05)
            symmetry = st.number_input('Symmetry', min_value=0.0, max_value=1.0, value=0.2)
            fractal_dimension = st.number_input('Fractal Dimension', min_value=0.0, max_value=1.0, value=0.06)
        
        if st.button('Predict'):
            input_data = np.array([[
                radius, texture, perimeter, area, smoothness,
                compactness, concavity, concave_points, symmetry, fractal_dimension,
                radius*0.1, texture*0.1, perimeter*0.1, area*0.1, smoothness*0.1,
                compactness*0.1, concavity*0.1, concave_points*0.1, symmetry*0.1, fractal_dimension*0.1,
                radius*1.2, texture*1.2, perimeter*1.2, area*1.2, smoothness*1.2,
                compactness*1.2, concavity*1.2, concave_points*1.2, symmetry*1.2, fractal_dimension*1.2
            ]])
            
            scaler = StandardScaler()
            X = processed_data.drop('diagnosis', axis=1)
            scaler.fit(X)
            input_scaled = scaler.transform(input_data)
            
            model = LogisticRegression()
            model.fit(X, processed_data['diagnosis'])
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            input_cluster = kmeans.predict(input_scaled)[0]
            
            st.subheader('Analysis Results')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Diagnosis Prediction")
                if prediction[0] == 1:
                    st.error('🚨 Malignant (M)')
                else:
                    st.success('✅ Benign (B)')
                st.info(f'Probability: {probability[0][1]:.2%} chance of being malignant')
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.subheader("Cluster Assignment")
                st.write(f"This case belongs to:")
                st.markdown(f"### Cluster {input_cluster}")
                
                cluster_labels = kmeans.predict(X)
                cluster_data = pd.DataFrame({
                    'cluster': cluster_labels,
                    'diagnosis': processed_data['diagnosis']
                })
                
                cluster_size = len(cluster_data[cluster_data['cluster'] == input_cluster])
                total_samples = len(cluster_data)
                st.write(f"📊 Cluster Size: {cluster_size} samples")
                st.write(f"📈 ({(cluster_size/total_samples):.1%} of total data)")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.subheader("Cluster Statistics")
                cluster_stats = cluster_data[cluster_data['cluster'] == input_cluster]['diagnosis'].value_counts(normalize=True)
                
                st.write("Cluster Composition:")
                if 1 in cluster_stats.index:
                    st.write(f"🔴 Malignant: {cluster_stats[1]:.1%}")
                if 0 in cluster_stats.index:
                    st.write(f"🟢 Benign: {cluster_stats[0]:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Diagram parameter (nampilin risiko M)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[0][1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Malignancy Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig)
            
            st.subheader("Detailed Cluster Analysis")
            
            cluster_means = X[cluster_labels == input_cluster].mean()
            cluster_std = X[cluster_labels == input_cluster].std()
            
            char_col1, char_col2 = st.columns(2)
            
            with char_col1:
                st.write("🔍 Key Characteristics of This Cluster:")
                top_features = cluster_means.nlargest(5)
                for feat, val in top_features.items():
                    st.write(f"- High **{feat}**: {val:.2f} ± {cluster_std[feat]:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with char_col2:
                st.write("📝 Input Values vs Cluster Average:")
                input_values = {
                    'radius_mean': radius,
                    'texture_mean': texture,
                    'perimeter_mean': perimeter
                }
                for feat in ['radius_mean', 'texture_mean', 'perimeter_mean']:
                    input_val = input_values.get(feat, 0)
                    cluster_val = cluster_means[feat]
                    diff = ((input_val - cluster_val) / cluster_val) * 100
                    st.write(f"- **{feat.replace('_mean', '')}**:")
                    st.write(f"  Input: {input_val:.2f}")
                    st.write(f"  Cluster avg: {cluster_val:.2f}")
                    st.write(f"  Difference: {diff:+.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("Model Evaluation Score")
            
            y = processed_data['diagnosis']
            
            model_accuracy = accuracy_score(y, model.predict(X))
            model_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
            
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            
            score_col1, score_col2 = st.columns(2)
                
            with score_col1:
                st.metric(
                    label="AUC Score",
                    value=f"{model_auc:.2%}",
                    help="Area Under the ROC Curve"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            with score_col2:
                st.metric(
                    label="Silhouette Score",
                    value=f"{silhouette_avg:.3f}",
                    help="Measure of cluster cohesion and separation (-1 to 1, higher is better)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
if __name__ == '__main__':
    main()
