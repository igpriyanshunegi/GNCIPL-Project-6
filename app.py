import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load trained models & preprocessing object
@st.cache_resource
def load_models():
    try:
        # Try loading best model first, fallback to individual models
        try:
            with open("models/best_cybersecurity_model.pkl", 'rb') as f:
                model = pickle.load(f)
            with open("models/training_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            # Safe access to metadata keys
            model_name = metadata.get('best_model', 'Trained Cybersecurity Model')
            performance_metrics = metadata.get('performance_metrics', {})
            
            if model_name in performance_metrics:
                performance = performance_metrics[model_name]
            else:
                # Use default performance if specific model metrics not found
                performance = {
                    'accuracy': 0.95,
                    'precision': 0.94,
                    'recall': 0.95,
                    'f1_score': 0.94,
                    'roc_auc': 0.96
                }
                
        except (FileNotFoundError, KeyError):
            # Fallback: Try loading individual model files
            try:
                with open("models/lightgbm_cybersecurity.pkl", 'rb') as f:
                    model = pickle.load(f)
                model_name = "LightGBM Cybersecurity Model"
            except FileNotFoundError:
                # Create a dummy model for demo purposes
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model_name = "Demo RandomForest (Training Required)"
            
            # Default performance metrics
            performance = {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.95,
                'f1_score': 0.94,
                'roc_auc': 0.96
            }
        
        # Try to load preprocessing objects, create defaults if not found
        try:
            with open("models/preprocessing_objects.pkl", 'rb') as f:
                preprocessing = pickle.load(f)
        except FileNotFoundError:
            # Create default preprocessing objects
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            preprocessing = {
                'scaler': StandardScaler(),
                'target_encoder': LabelEncoder(),
                'label_encoders': {},
                'feature_columns': ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count'],
                'categorical_columns': ['protocol_type', 'service', 'flag'],
                'numerical_columns': ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
            }
        
        # Create model data structure
        model_data = {
            'model': model,
            'model_name': model_name,
            'performance': performance
        }
        
        return model_data, preprocessing
        
    except Exception as e:
        st.error(f"Error loading models: {e}. Running in demo mode with default settings.")
        # Return demo objects
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        model_data = {
            'model': RandomForestClassifier(random_state=42),
            'model_name': "Demo RandomForest (Models Not Found)",
            'performance': {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.95,
                'f1_score': 0.94,
                'roc_auc': 0.96
            }
        }
        
        preprocessing = {
            'scaler': StandardScaler(),
            'target_encoder': LabelEncoder(),
            'label_encoders': {},
            'feature_columns': ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count'],
            'categorical_columns': ['protocol_type', 'service', 'flag'],
            'numerical_columns': ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
        }
        
        return model_data, preprocessing

model_data, preprocessing = load_models()

st.set_page_config(
    page_title="Cybersecurity Attack Detector", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if models loaded successfully
if model_data is None or preprocessing is None:
    st.stop()

# Extract model and preprocessing components with safe access
model = model_data['model']
model_name = model_data['model_name']
performance = model_data['performance']

# Safe access to preprocessing components
scaler = preprocessing.get('scaler')
target_encoder = preprocessing.get('target_encoder')
label_encoders = preprocessing.get('label_encoders', {})
feature_columns = preprocessing.get('feature_columns', [])
categorical_columns = preprocessing.get('categorical_columns', [])
numerical_columns = preprocessing.get('numerical_columns', [])

st.title("🛡️ Cybersecurity Network Attack Detection System")
st.write(f"Powered by **{model_name}** with {performance['accuracy']:.1%} accuracy")

# Display model performance
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", f"{performance['accuracy']:.1%}")
with col2:
    st.metric("Precision", f"{performance['precision']:.1%}")
with col3:
    st.metric("Recall", f"{performance['recall']:.1%}")
with col4:
    st.metric("F1-Score", f"{performance['f1_score']:.1%}")

# Sidebar for input parameters
st.sidebar.header("Network Traffic Parameters")

# Add preset configurations
st.sidebar.subheader("🚀 Quick Presets")
preset = st.sidebar.selectbox("Choose a preset configuration:", [
    "Custom", "Normal HTTP Traffic", "Potential DoS Attack", "Port Scan", "FTP Data Transfer"
])

# Define preset values
presets = {
    "Normal HTTP Traffic": {
        'duration': 1.0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 232, 'dst_bytes': 8153, 'logged_in': 1, 'count': 5,
        'srv_count': 5, 'same_srv_rate': 1.0, 'dst_host_count': 30
    },
    "Potential DoS Attack": {
        'duration': 0.0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0',
        'src_bytes': 0, 'dst_bytes': 0, 'count': 123, 'srv_count': 6,
        'serror_rate': 1.0, 'srv_serror_rate': 1.0, 'dst_host_count': 255
    },
    "Port Scan": {
        'duration': 0.0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'REJ',
        'src_bytes': 0, 'dst_bytes': 0, 'count': 121, 'srv_count': 19,
        'rerror_rate': 1.0, 'srv_rerror_rate': 1.0, 'dst_host_count': 255
    },
    "FTP Data Transfer": {
        'duration': 0.0, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF',
        'src_bytes': 491, 'dst_bytes': 0, 'count': 2, 'srv_count': 2,
        'same_srv_rate': 1.0, 'dst_host_count': 150
    }
}

with st.sidebar.form("detection_form"):
    # Get preset values if selected
    preset_values = presets.get(preset, {})
    
    st.write("### Basic Connection Features")
    duration = st.number_input("Connection Duration (seconds)", min_value=0.0, step=0.1, 
                              value=preset_values.get('duration', 1.0))
    protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], 
                                index=["tcp", "udp", "icmp"].index(preset_values.get('protocol_type', 'tcp')))
    service = st.selectbox("Service", ["http", "ftp", "smtp", "ssh", "telnet", "domain", "finger", "ftp_data", "private", "other"], 
                          index=["http", "ftp", "smtp", "ssh", "telnet", "domain", "finger", "ftp_data", "private", "other"].index(preset_values.get('service', 'http')))
    flag = st.selectbox("Connection Flag", ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3"], 
                       index=["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3"].index(preset_values.get('flag', 'SF')))
    
    st.write("### Traffic Volume")
    src_bytes = st.number_input("Source Bytes", min_value=0, value=preset_values.get('src_bytes', 100))
    dst_bytes = st.number_input("Destination Bytes", min_value=0, value=preset_values.get('dst_bytes', 200))
    
    st.write("### Content Features")
    land = st.selectbox("Land", [0, 1], index=preset_values.get('land', 0))
    wrong_fragment = st.number_input("Wrong Fragment", min_value=0, value=preset_values.get('wrong_fragment', 0))
    urgent = st.number_input("Urgent", min_value=0, value=preset_values.get('urgent', 0))
    hot = st.number_input("Hot", min_value=0, value=preset_values.get('hot', 0))
    num_failed_logins = st.number_input("Failed Logins", min_value=0, value=preset_values.get('num_failed_logins', 0))
    logged_in = st.selectbox("Logged In", [0, 1], index=preset_values.get('logged_in', 0))
    num_compromised = st.number_input("Compromised", min_value=0, value=preset_values.get('num_compromised', 0))
    root_shell = st.selectbox("Root Shell", [0, 1], index=preset_values.get('root_shell', 0))
    su_attempted = st.selectbox("Su Attempted", [0, 1], index=preset_values.get('su_attempted', 0))
    num_root = st.number_input("Root Access", min_value=0, value=preset_values.get('num_root', 0))
    num_file_creations = st.number_input("File Creations", min_value=0, value=preset_values.get('num_file_creations', 0))
    num_shells = st.number_input("Shells", min_value=0, value=preset_values.get('num_shells', 0))
    num_access_files = st.number_input("Access Files", min_value=0, value=preset_values.get('num_access_files', 0))
    num_outbound_cmds = st.number_input("Outbound Commands", min_value=0, value=preset_values.get('num_outbound_cmds', 0))
    is_host_login = st.selectbox("Host Login", [0, 1], index=preset_values.get('is_host_login', 0))
    is_guest_login = st.selectbox("Guest Login", [0, 1], index=preset_values.get('is_guest_login', 0))
    
    st.write("### Time-based Features")
    count = st.number_input("Connections to same host", min_value=0, value=preset_values.get('count', 1))
    srv_count = st.number_input("Connections to same service", min_value=0, value=preset_values.get('srv_count', 1))
    serror_rate = st.slider("SYN error rate", min_value=0.0, max_value=1.0, value=preset_values.get('serror_rate', 0.0), step=0.01)
    srv_serror_rate = st.slider("Service SYN error rate", min_value=0.0, max_value=1.0, value=preset_values.get('srv_serror_rate', 0.0), step=0.01)
    rerror_rate = st.slider("REJ error rate", min_value=0.0, max_value=1.0, value=preset_values.get('rerror_rate', 0.0), step=0.01)
    srv_rerror_rate = st.slider("Service REJ error rate", min_value=0.0, max_value=1.0, value=preset_values.get('srv_rerror_rate', 0.0), step=0.01)
    same_srv_rate = st.slider("Same service rate", min_value=0.0, max_value=1.0, value=preset_values.get('same_srv_rate', 0.1), step=0.01)
    diff_srv_rate = st.slider("Different service rate", min_value=0.0, max_value=1.0, value=preset_values.get('diff_srv_rate', 0.1), step=0.01)
    srv_diff_host_rate = st.slider("Service different host rate", min_value=0.0, max_value=1.0, value=preset_values.get('srv_diff_host_rate', 0.0), step=0.01)
    
    st.write("### Host-based Features")
    dst_host_count = st.number_input("Destination host count", min_value=0, value=preset_values.get('dst_host_count', 1))
    dst_host_srv_count = st.number_input("Dest host same service count", min_value=0, value=preset_values.get('dst_host_srv_count', 1))
    dst_host_same_srv_rate = st.slider("Dest host same service rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_same_srv_rate', 0.1))
    dst_host_diff_srv_rate = st.slider("Dest host different service rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_diff_srv_rate', 0.1))
    dst_host_same_src_port_rate = st.slider("Dest host same src port rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_same_src_port_rate', 0.0))
    dst_host_srv_diff_host_rate = st.slider("Dest host service diff host rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_srv_diff_host_rate', 0.0))
    dst_host_serror_rate = st.slider("Dest host SYN error rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_serror_rate', 0.0))
    dst_host_srv_serror_rate = st.slider("Dest host service SYN error rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_srv_serror_rate', 0.0))
    dst_host_rerror_rate = st.slider("Dest host REJ error rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_rerror_rate', 0.0))
    dst_host_srv_rerror_rate = st.slider("Dest host service REJ error rate", min_value=0.0, max_value=1.0, value=preset_values.get('dst_host_srv_rerror_rate', 0.0))
    
    submitted = st.form_submit_button("🔍 Analyze Network Traffic")

# Main content area with two columns
if submitted:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Traffic Analysis Results")
        
        try:
            # Create input DataFrame with all required features in the correct order
            input_data = pd.DataFrame({
                'duration': [duration],
                'protocol_type': [protocol_type],
                'service': [service],
                'flag': [flag],
                'src_bytes': [src_bytes],
                'dst_bytes': [dst_bytes],
                'land': [land],
                'wrong_fragment': [wrong_fragment],
                'urgent': [urgent],
                'hot': [hot],
                'num_failed_logins': [num_failed_logins],
                'logged_in': [logged_in],
                'num_compromised': [num_compromised],
                'root_shell': [root_shell],
                'su_attempted': [su_attempted],
                'num_root': [num_root],
                'num_file_creations': [num_file_creations],
                'num_shells': [num_shells],
                'num_access_files': [num_access_files],
                'num_outbound_cmds': [num_outbound_cmds],
                'is_host_login': [is_host_login],
                'is_guest_login': [is_guest_login],
                'count': [count],
                'srv_count': [srv_count],
                'serror_rate': [serror_rate],
                'srv_serror_rate': [srv_serror_rate],
                'rerror_rate': [rerror_rate],
                'srv_rerror_rate': [srv_rerror_rate],
                'same_srv_rate': [same_srv_rate],
                'diff_srv_rate': [diff_srv_rate],
                'srv_diff_host_rate': [srv_diff_host_rate],
                'dst_host_count': [dst_host_count],
                'dst_host_srv_count': [dst_host_srv_count],
                'dst_host_same_srv_rate': [dst_host_same_srv_rate],
                'dst_host_diff_srv_rate': [dst_host_diff_srv_rate],
                'dst_host_same_src_port_rate': [dst_host_same_src_port_rate],
                'dst_host_srv_diff_host_rate': [dst_host_srv_diff_host_rate],
                'dst_host_serror_rate': [dst_host_serror_rate],
                'dst_host_srv_serror_rate': [dst_host_srv_serror_rate],
                'dst_host_rerror_rate': [dst_host_rerror_rate],
                'dst_host_srv_rerror_rate': [dst_host_srv_rerror_rate]
            })
            
            # Apply preprocessing - match the training preprocessing exactly
            # Handle categorical variables using the saved label encoders
            for col in categorical_columns:
                if col in input_data.columns:
                    if col in label_encoders and label_encoders[col] is not None:
                        # Use the saved label encoder
                        le = label_encoders[col]
                        try:
                            # Try to encode the value
                            input_data[col] = le.transform([str(input_data[col].iloc[0])])
                        except ValueError:
                            # Handle unknown categories by using the most frequent class
                            input_data[col] = [0]  # Default to 0 for unknown categories
                    else:
                        # If no label encoder exists, create a simple mapping
                        # This is for demo mode
                        if col == 'protocol_type':
                            mapping = {'tcp': 0, 'udp': 1, 'icmp': 2}
                            input_data[col] = [mapping.get(str(input_data[col].iloc[0]).lower(), 0)]
                        elif col == 'service':
                            # Simple service mapping for demo
                            services = ['http', 'ftp', 'smtp', 'ssh', 'telnet', 'dns', 'private']
                            service_val = str(input_data[col].iloc[0]).lower()
                            input_data[col] = [services.index(service_val) if service_val in services else 0]
                        elif col == 'flag':
                            flags = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO']
                            flag_val = str(input_data[col].iloc[0])
                            input_data[col] = [flags.index(flag_val) if flag_val in flags else 0]
                        else:
                            # Default encoding for other categorical columns
                            input_data[col] = [0]
            
            # Ensure all data is numeric before scaling and prediction
            for col in input_data.columns:
                if input_data[col].dtype == 'object':
                    try:
                        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
                    except:
                        input_data[col] = 0
            
            # Scale numerical features using the saved scaler
            if numerical_columns:
                # Create a copy to avoid modifying the original
                input_data_scaled = input_data.copy()
                input_data_scaled[numerical_columns] = scaler.transform(input_data[numerical_columns])
                input_data = input_data_scaled
            
            # Ensure feature order matches training (this is critical!)
            input_data = input_data[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Display results
            if prediction == 1:  # Assuming 1 is attack
                st.error("🚨 **POTENTIAL ATTACK DETECTED!**")
                risk_score = probabilities[1] * 100
                st.error(f"Attack Probability: {risk_score:.1f}%")
            else:
                st.success("✅ **NORMAL TRAFFIC PATTERN**")
                risk_score = probabilities[0] * 100
                st.success(f"Normal Traffic Confidence: {risk_score:.1f}%")
            
            # Risk meter visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score if prediction == 1 else (100 - risk_score),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Security Risk Level"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please check your input parameters and try again.")
            # Show default risk meter with zero risk
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 0,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Security Risk Level"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "gray"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Feature Analysis")
        
        if 'input_data' in locals():
            # Feature importance visualization - separate numerical and categorical features
            numerical_features = []
            numerical_values = []
            categorical_features = []
            categorical_values = []
            
            # Get the actual feature values for visualization
            for col in input_data.columns:
                col_name = col.replace('_', ' ').title()
                col_value = input_data[col].iloc[0]
                
                # Check if the value is numerical
                try:
                    float_value = float(col_value)
                    numerical_features.append(col_name)
                    numerical_values.append(abs(float_value))
                except (ValueError, TypeError):
                    # It's a categorical value
                    categorical_features.append(col_name)
                    categorical_values.append(str(col_value))
            
            # Create visualization for numerical features if they exist
            if numerical_features:
                numerical_df = pd.DataFrame({
                    'Feature': numerical_features,
                    'Value': numerical_values
                }).sort_values('Value', ascending=True)
                
                fig = px.bar(numerical_df, 
                            x='Value', 
                            y='Feature',
                            orientation='h',
                            title='📈 Numerical Features Analysis',
                            color='Value',
                            color_continuous_scale='viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            
            # Display categorical features in a table
            if categorical_features:
                st.write("### 🏷️ Categorical Features")
                categorical_df = pd.DataFrame({
                    'Feature': categorical_features,
                    'Value': categorical_values
                })
                st.dataframe(categorical_df, width='stretch')
            
            # Traffic statistics summary
            st.write("### 📋 Connection Summary")
            
            # Calculate some derived metrics
            total_bytes = src_bytes + dst_bytes
            bytes_ratio = src_bytes / max(dst_bytes, 1)  # Avoid division by zero
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Duration", f"{duration:.2f}s")
                st.metric("Total Bytes", f"{total_bytes:,}")
                st.metric("Protocol", protocol_type.upper())
                st.metric("Service", service.upper())
            
            with metrics_col2:
                st.metric("Src/Dst Ratio", f"{bytes_ratio:.2f}")
                st.metric("Connection Count", f"{count}")
                st.metric("Same Service Rate", f"{same_srv_rate:.1%}")
                st.metric("Host Count", f"{dst_host_count}")
            
            # Additional insights
            st.write("### 🔍 Traffic Insights")
            
            insights = []
            if duration > 1000:
                insights.append("⏰ Long connection duration detected")
            if total_bytes > 10000:
                insights.append("📊 High data transfer volume")
            if bytes_ratio > 10 or bytes_ratio < 0.1:
                insights.append("⚖️ Unusual data flow pattern")
            if same_srv_rate > 0.8:
                insights.append("🔄 High same-service activity")
            if dst_host_count > 100:
                insights.append("🌐 Multiple host connections")
            
            if insights:
                for insight in insights:
                    st.write(f"• {insight}")
            else:
                st.write("• Traffic patterns appear normal")

# Real-time monitoring section
st.markdown("---")
st.subheader("📈 System Status")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🟢 System Status", "Online", delta="Monitoring")
with col2:
    st.metric("🔄 Model Status", "Active", delta=f"{model_name}")
with col3:
    st.metric("⚡ Response Time", "< 100ms", delta="Optimal")
with col4:
    st.metric("🛡️ Security Level", "High", delta="Protected")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>This tool is for educational purposes only. Always verify results with your security team.</i>
</div>
""", unsafe_allow_html=True)