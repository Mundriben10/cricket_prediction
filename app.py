import streamlit as st
import pandas as pd
import numpy as np
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🏏 Cricket Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E8B57;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🏏 Cricket Win Prediction System</h1>', unsafe_allow_html=True)

# Initialize session state variables
if 'models' not in st.session_state:
    st.session_state.models = None
if 'scores' not in st.session_state:
    st.session_state.scores = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Auto-load pre-trained models on startup
@st.cache_resource
def auto_load_models():
    """Automatically load pre-trained models if available"""
    try:
        # Try to load from different possible locations
        model_files = ['cricket_models.pkl', 'models/cricket_models.pkl', 'saved_models.pkl']
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                st.success(f"✅ Pre-trained models loaded successfully from {model_file}!")
                return model_data
                
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Error loading {model_file}: {e}")
                continue
        
        return None
        
    except Exception as e:
        st.error(f"❌ Error in auto-loading models: {e}")
        return None

# Load models automatically if not already loaded
if not st.session_state.models_loaded:
    model_data = auto_load_models()
    if model_data:
        st.session_state.models = model_data.get('models')
        st.session_state.encoders = model_data.get('encoders')
        st.session_state.scaler = model_data.get('scaler')
        st.session_state.scores = model_data.get('scores', {})
        st.session_state.models_loaded = True
        
        # Show success message in sidebar
        st.sidebar.success("🎯 Models Ready!")
        st.sidebar.info(f"📊 {len(st.session_state.models)} models loaded")
    else:
        st.sidebar.warning("⚠️ No pre-trained models found")
        st.sidebar.info("👆 Train models once in 'Model Training' page")

# Sidebar for navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Choose a page:", 
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Prediction", "📈 Model Comparison"])

@st.cache_data
def load_data():
    """Load and preprocess the cricket data"""
    try:
        # Try different possible file names
        possible_files = ['final_data.csv', 'cricket_data.csv', 'data.csv']
        
        for filename in possible_files:
            try:
                data = pd.read_csv(filename)
                # Drop the unnamed index column if it exists
                columns_to_drop = [col for col in data.columns if 'Unnamed' in str(col) or col == '']
                if columns_to_drop:
                    data = data.drop(columns_to_drop, axis=1)
                
                # Validate required columns
                required_columns = ['batting_team', 'bowling_team', 'city', 'runs_left', 
                                  'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result']
                
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.write("Available columns:", list(data.columns))
                    return None
                
                st.success(f"✅ Data loaded successfully from {filename}")
                return data
                
            except FileNotFoundError:
                continue
        
        st.error("❌ Cricket data file not found. Please ensure 'final_data.csv' is in the app directory.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    try:
        # Create label encoders for categorical variables
        le_batting = LabelEncoder()
        le_bowling = LabelEncoder()
        le_city = LabelEncoder()
        
        # Encode categorical variables
        df_processed = df.copy()
        df_processed['batting_team_encoded'] = le_batting.fit_transform(df['batting_team'])
        df_processed['bowling_team_encoded'] = le_bowling.fit_transform(df['bowling_team'])
        df_processed['city_encoded'] = le_city.fit_transform(df['city'])
        
        # Select features for modeling
        feature_cols = ['batting_team_encoded', 'bowling_team_encoded', 'city_encoded', 
                       'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']
        
        # Check if all feature columns exist
        missing_features = [col for col in feature_cols if col not in df_processed.columns]
        if missing_features:
            st.error(f"Missing features after encoding: {missing_features}")
            return None, None, None, None, None
        
        X = df_processed[feature_cols]
        y = df_processed['result']
        
        # Validate data types
        X = X.fillna(0)  # Handle any NaN values
        y = y.fillna(0)  # Handle any NaN values
        
        return X, y, le_batting, le_bowling, le_city
        
    except Exception as e:
        st.error(f"❌ Error preprocessing data: {str(e)}")
        return None, None, None, None, None

def train_models(X_train, y_train):
    """Train multiple machine learning models with detailed progress tracking"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    trained_models = {}
    model_scores = {}
    
    # Create containers for progress display
    progress_container = st.container()
    
    with progress_container:
        # Overall progress
        st.subheader("🔄 Training Progress")
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        # Individual model progress table
        st.subheader("📊 Model Training Status")
        
        # Create placeholder for results table
        results_placeholder = st.empty()
        
        # Initialize results data
        results_data = []
        for name in models.keys():
            results_data.append({
                'Model': name,
                'Status': '⏳ Pending',
                'Progress': '0%',
                'CV Score': '-',
                'Std Dev': '-'
            })
        
        # Display initial table
        results_df = pd.DataFrame(results_data)
        results_placeholder.dataframe(results_df, use_container_width=True)
    
    # Train each model
    for i, (name, model) in enumerate(models.items()):
        # Update overall progress
        overall_percentage = int((i / len(models)) * 100)
        overall_progress.progress(i / len(models))
        overall_status.markdown(f"**Training: {name}** | Overall Progress: {overall_percentage}%")
        
        # Update current model status to "Training"
        results_data[i]['Status'] = '🔄 Training...'
        results_data[i]['Progress'] = f'{overall_percentage}%'
        results_df = pd.DataFrame(results_data)
        results_placeholder.dataframe(results_df, use_container_width=True)
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            model_scores[name] = {
                'mean_cv_score': mean_score,
                'std_cv_score': std_score
            }
            
            # Update model status to "Completed"
            results_data[i]['Status'] = '✅ Completed'
            results_data[i]['Progress'] = '100%'
            results_data[i]['CV Score'] = f'{mean_score:.4f}'
            results_data[i]['Std Dev'] = f'±{std_score:.4f}'
            
        except Exception as e:
            # Handle training errors
            results_data[i]['Status'] = '❌ Failed'
            results_data[i]['Progress'] = 'Error'
            results_data[i]['CV Score'] = 'Error'
            results_data[i]['Std Dev'] = str(e)[:20] + '...'
            st.error(f"Error training {name}: {e}")
        
        # Update the results table
        results_df = pd.DataFrame(results_data)
        results_placeholder.dataframe(results_df, use_container_width=True)
    
    # Final progress update
    overall_progress.progress(1.0)
    overall_status.markdown("**🎉 Training Completed!** | Overall Progress: 100%")
    
    # Show final summary
    st.success("✅ All models trained successfully!")
    
    # Display best performing model
    if model_scores:
        best_model = max(model_scores.items(), key=lambda x: x[1]['mean_cv_score'])
        st.info(f"🏆 **Best Model**: {best_model[0]} with CV Score: {best_model[1]['mean_cv_score']:.4f}")
    
    # Auto-save trained models
    try:
        model_data = {
            'models': trained_models,
            'scores': model_scores,
            'encoders': st.session_state.encoders,
            'scaler': st.session_state.scaler
        }
        
        with open('cricket_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        st.success("💾 Models automatically saved for future use!")
        
    except Exception as e:
        st.warning(f"⚠️ Could not auto-save models: {e}")
    
    return trained_models, model_scores

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model"""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Check if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        return metrics, y_pred, y_pred_proba
        
    except Exception as e:
        st.error(f"❌ Error evaluating model: {str(e)}")
        return None, None, None

# Home Page
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to Cricket Win Prediction System!</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>State-of-the-art ML models for precise win probability estimation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Comprehensive Analysis</h3>
            <p>Deep insights into team performance and match dynamics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Real-time Results</h3>
            <p>Instant predictions with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏏 About This Project")
    st.write("""
    This Cricket Win Prediction System uses advanced machine learning techniques to predict match outcomes 
    based on current match situation. The system analyzes various factors including:
    
    - **Team Performance**: Historical performance of batting and bowling teams
    - **Match Context**: Current runs needed, balls remaining, wickets in hand
    - **Venue Impact**: City-wise performance analysis
    - **Real-time Metrics**: Current run rate vs required run rate analysis
    
    Navigate through different sections to explore data insights, train models, and make predictions!
    """)

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None:
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", f"{len(data):,}")
        with col2:
            st.metric("Total Teams", len(data['batting_team'].unique()))
        with col3:
            st.metric("Total Cities", len(data['city'].unique()))
        with col4:
            win_rate = (data['result'].sum() / len(data)) * 100
            st.metric("Overall Win Rate", f"{win_rate:.1f}%")
        
        st.markdown("---")
        
        # Data visualization
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Team Analysis", "🏟️ Venue Analysis", "📈 Match Dynamics", "🔍 Feature Distribution"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Team win rates as batting team
                team_wins = data.groupby('batting_team')['result'].agg(['sum', 'count']).reset_index()
                team_wins['win_rate'] = (team_wins['sum'] / team_wins['count']) * 100
                team_wins = team_wins.sort_values('win_rate', ascending=False)
                
                fig = px.bar(team_wins, x='win_rate', y='batting_team', 
                           title='Team Win Rates (as Batting Team)',
                           labels={'win_rate': 'Win Rate (%)', 'batting_team': 'Team'},
                           color='win_rate',
                           color_continuous_scale='RdYlGn')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Total matches per team
                batting_counts = data['batting_team'].value_counts()
                bowling_counts = data['bowling_team'].value_counts()
                total_matches = batting_counts + bowling_counts
                
                fig = px.pie(values=total_matches.values, names=total_matches.index,
                           title='Total Matches per Team')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # City-wise match distribution
                city_matches = data['city'].value_counts().head(15)
                fig = px.bar(x=city_matches.values, y=city_matches.index,
                           title='Top 15 Cities by Number of Matches',
                           labels={'x': 'Number of Matches', 'y': 'City'},
                           color=city_matches.values,
                           color_continuous_scale='viridis')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # City-wise win rates
                city_wins = data.groupby('city')['result'].agg(['sum', 'count']).reset_index()
                city_wins['win_rate'] = (city_wins['sum'] / city_wins['count']) * 100
                city_wins = city_wins[city_wins['count'] >= 10]  # Filter cities with at least 10 matches
                city_wins = city_wins.sort_values('win_rate', ascending=False)
                
                fig = px.scatter(city_wins, x='count', y='win_rate', 
                               hover_data=['city'],
                               title='Win Rate vs Number of Matches by City',
                               labels={'count': 'Number of Matches', 'win_rate': 'Win Rate (%)'},
                               color='win_rate',
                               color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Runs left vs Win probability
                bins = np.arange(0, min(data['runs_left'].max() + 20, 200), 20)
                data['runs_left_bin'] = pd.cut(data['runs_left'], bins)
                runs_analysis = data.groupby('runs_left_bin')['result'].agg(['mean', 'count']).reset_index()
                runs_analysis = runs_analysis[runs_analysis['count'] >= 10]
                runs_analysis['runs_left_mid'] = runs_analysis['runs_left_bin'].apply(lambda x: x.mid if pd.notna(x) else 0)
                
                fig = px.line(runs_analysis, x='runs_left_mid', y='mean',
                            title='Win Probability vs Runs Remaining',
                            labels={'runs_left_mid': 'Runs Remaining', 'mean': 'Win Probability'})
                fig.update_traces(line_color='red', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Required run rate vs Win probability
                data['rrr_bin'] = pd.cut(data['rrr'], bins=np.arange(0, min(data['rrr'].max() + 1, 20), 1))
                rrr_analysis = data.groupby('rrr_bin')['result'].agg(['mean', 'count']).reset_index()
                rrr_analysis = rrr_analysis[rrr_analysis['count'] >= 10]
                rrr_analysis['rrr_mid'] = rrr_analysis['rrr_bin'].apply(lambda x: x.mid if pd.notna(x) else 0)
                
                fig = px.line(rrr_analysis, x='rrr_mid', y='mean',
                            title='Win Probability vs Required Run Rate',
                            labels={'rrr_mid': 'Required Run Rate', 'mean': 'Win Probability'})
                fig.update_traces(line_color='blue', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Feature distributions
            numeric_features = ['runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']
            
            for i in range(0, len(numeric_features), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(numeric_features):
                        feature = numeric_features[i]
                        fig = px.histogram(data, x=feature, color='result',
                                         title=f'Distribution of {feature}',
                                         marginal='box')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if i + 1 < len(numeric_features):
                        feature = numeric_features[i + 1]
                        fig = px.histogram(data, x=feature, color='result',
                                         title=f'Distribution of {feature}',
                                         marginal='box')
                        st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    # Check if models are already trained
    if st.session_state.models is not None:
        st.success("✅ Models are already trained and ready to use!")
        
        # Display current model status
        st.subheader("📊 Current Model Status")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Models Available", f"{len(st.session_state.models)}")
        with col2:
            if st.session_state.scores:
                best_score = max(st.session_state.scores.values(), key=lambda x: x.get('mean_cv_score', 0))
                st.metric("🏆 Best CV Score", f"{best_score.get('mean_cv_score', 0):.4f}")
            else:
                st.metric("🏆 Best CV Score", "N/A")
        with col3:
            st.metric("📊 Status", "Ready")
        with col4:
            st.metric("🎯 Action", "Use Models")
        
        # Show model list
        st.subheader("🤖 Available Models")
        model_list = list(st.session_state.models.keys())
        
        # Create a nice display of models
        cols = st.columns(3)
        for i, model_name in enumerate(model_list):
            with cols[i % 3]:
                score = "N/A"
                if st.session_state.scores and model_name in st.session_state.scores:
                    score = f"{st.session_state.scores[model_name]['mean_cv_score']:.4f}"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>✅ {model_name}</h4>
                    <p>CV Score: {score}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Option to retrain if needed
        st.subheader("🔄 Retrain Models (Optional)")
        st.warning("⚠️ **Note**: Models are already trained. Retraining will overwrite existing models.")
        
        with st.expander("🔧 Advanced: Retrain Models"):
            st.info("Only retrain if you want to experiment with different parameters or have new data.")
            
            data = load_data()
            if data is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Size", f"{len(data):,} records")
                with col2:
                    st.metric("Features", "9 features")
                with col3:
                    test_size = st.selectbox("Test Size", [0.2, 0.25, 0.3], index=0)
                
                if st.button("🚀 Start Model Training", key="train_button"):
                    with st.spinner("Preparing data and initializing models..."):
                        # Preprocess data
                        X, y, le_batting, le_bowling, le_city = preprocess_data(data)
                        
                        if X is None:
                            st.error("Failed to preprocess data. Please check your dataset.")
                            st.stop()
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Store in session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.scaler = scaler
                        st.session_state.encoders = (le_batting, le_bowling, le_city)
                        
                        # Display data split information
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training Set", f"{len(X_train):,}")
                        with col2:
                            st.metric("Test Set", f"{len(X_test):,}")
                        with col3:
                            st.metric("Training Win Rate", f"{y_train.mean():.2%}")
                        with col4:
                            st.metric("Test Win Rate", f"{y_test.mean():.2%}")
                    
                    # Train models with detailed progress
                    models, scores = train_models(X_train, y_train)
                    st.session_state.models = models
                    st.session_state.scores = scores
                    st.session_state.models_loaded = True
                    
                    st.success("🎉 **Training Complete!** You can now make predictions on the 'Prediction' page.")
            
            else:
                # Show training preview
                st.info("""
                🚀 **Ready to train models!**
                
                This will train 6 different machine learning models:
                - Logistic Regression
                - Random Forest  
                - Gradient Boosting
                - Decision Tree
                - K-Nearest Neighbors
                - Naive Bayes
                
                Each model will be evaluated using 5-fold cross-validation.
                Models will be automatically saved for future use.
                """)