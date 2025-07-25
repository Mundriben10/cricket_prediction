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
    
    else:
        # No models found - show training interface
        st.info("🚀 **No trained models found.** Let's train some models!")
        
        data = load_data()
        if data is not None:
            # Training configuration section
            st.subheader("⚙️ Training Configuration")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset Size", f"{len(data):,} records")
            with col2:
                st.metric("Features", "9 features")
            with col3:
                test_size = st.selectbox("Test Size", [0.2, 0.25, 0.3], index=0)
            
            st.markdown("---")
            
            if st.button("🚀 Start Model Training", key="main_train_button", type="primary"):
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
            st.error("❌ Unable to load the dataset. Please ensure 'final_data.csv' is in the correct location.")

# Prediction Page
elif page == "🔮 Prediction":
    st.markdown('<h2 class="sub-header">🔮 Match Win Prediction</h2>', unsafe_allow_html=True)
    
    # Check if models are available
    if st.session_state.models is None:
        st.error("❌ **No models available for prediction!**")
        st.info("""
        **What to do:**
        1. 🏠 If this is your first time: Go to **Model Training** page to train models (one-time setup)
        2. 📁 If you have saved models: Use the **Load Saved Models** option in the sidebar
        3. 🔄 If models failed to load: Check if 'cricket_models.pkl' exists in the app directory
        """)
        
        # Quick load option
        st.markdown("---")
        st.subheader("🚀 Quick Setup")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏃‍♂️ Go to Model Training", key="goto_training"):
                st.info("👆 Please use the navigation menu to go to 'Model Training' page")
        
        with col2:
            st.info("📁 Or upload your saved models using the sidebar")
    
    else:
        # Models available - show prediction interface
        data = load_data()
        if data is not None:
            # Success message
            st.success(f"✅ **Ready for Predictions!** {len(st.session_state.models)} models loaded.")
            
            # Create prediction interface
            st.subheader("⚙️ Match Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                batting_team = st.selectbox("🏏 Batting Team", sorted(data['batting_team'].unique()))
                bowling_team = st.selectbox("🥎 Bowling Team", sorted(data['bowling_team'].unique()))
                city = st.selectbox("🏟️ City", sorted(data['city'].unique()))
                
            with col2:
                runs_left = st.number_input("🎯 Runs Needed", min_value=1, max_value=300, value=50)
                balls_left = st.number_input("⏰ Balls Remaining", min_value=1, max_value=120, value=30)
                wickets = st.number_input("🏏 Wickets in Hand", min_value=1, max_value=10, value=6)
            
            col3, col4 = st.columns(2)
            with col3:
                total_runs = st.number_input("📊 Total Target", min_value=50, max_value=300, value=180)
                
            with col4:
                # Calculate current run rate and required run rate
                balls_bowled = 120 - balls_left  # Assuming T20 format
                if balls_bowled > 0:
                    current_score = total_runs - runs_left
                    crr = (current_score / balls_bowled) * 6
                else:
                    crr = 0.0
                
                rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 0.0
                
                st.metric("Current Run Rate", f"{crr:.2f}")
                st.metric("Required Run Rate", f"{rrr:.2f}")
            
            # Prediction button
            st.markdown("---")
            if st.button("🎯 Predict Win Probability", key="predict_button", type="primary"):
                # Prepare input data
                try:
                    le_batting, le_bowling, le_city = st.session_state.encoders
                    
                    # Check if team/city exists in training data
                    try:
                        batting_encoded = le_batting.transform([batting_team])[0]
                        bowling_encoded = le_bowling.transform([bowling_team])[0]
                        city_encoded = le_city.transform([city])[0]
                    except ValueError as e:
                        st.error(f"❌ Error: {batting_team}, {bowling_team}, or {city} was not seen during training. Please select different teams/city.")
                        st.stop()
                    
                    input_data = np.array([[batting_encoded, bowling_encoded, city_encoded, 
                                          runs_left, balls_left, wickets, total_runs, crr, rrr]])
                    
                    # Get predictions from all models
                    predictions = {}
                    probabilities = {}
                    
                    for name, model in st.session_state.models.items():
                        try:
                            pred = model.predict(input_data)[0]
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(input_data)[0][1]  # Probability of winning
                            else:
                                prob = pred  # For models without predict_proba
                            
                            predictions[name] = pred
                            probabilities[name] = prob
                        except Exception as e:
                            st.warning(f"⚠️ Error with {name}: {str(e)}")
                            continue
                    
                    if not predictions:
                        st.error("❌ No models could make predictions. Please retrain the models.")
                    else:
                        # Ensemble prediction (voting)
                        ensemble_prob = np.mean(list(probabilities.values()))
                        ensemble_pred = 1 if ensemble_prob > 0.5 else 0
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("🎯 Prediction Results")
                        
                        # Main prediction card
                        result_text = "WIN" if ensemble_pred == 1 else "LOSE"
                        confidence = max(ensemble_prob, 1 - ensemble_prob) * 100
                        
                        # Color coding based on probability
                        if ensemble_prob >= 0.7:
                            card_color = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"  # Green
                        elif ensemble_prob >= 0.3:
                            card_color = "linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)"  # Orange
                        else:
                            card_color = "linear-gradient(135deg, #dc3545 0%, #e83e8c 100%)"  # Red
                        
                        st.markdown(f"""
                        <div style="background: {card_color}; padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <h2>🏆 Predicted Outcome: {result_text}</h2>
                            <h3>🎯 Win Probability: {ensemble_prob:.2%}</h3>
                            <h4>📊 Confidence Level: {confidence:.1f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Match situation analysis
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 Run Rate Gap", f"{rrr - crr:.2f}", delta=f"{rrr - crr:.2f}")
                        with col2:
                            st.metric("⏰ Pressure Index", f"{(rrr/crr)*100 if crr > 0 else 999:.0f}%")
                        with col3:
                            wickets_factor = wickets / 10
                            st.metric("🏏 Wickets Factor", f"{wickets_factor:.1f}", delta=f"{wickets}")
                        
                        # Individual model predictions
                        st.subheader("🤖 Individual Model Predictions")
                        
                        model_results = pd.DataFrame({
                            'Model': list(predictions.keys()),
                            'Prediction': ['WIN' if p == 1 else 'LOSE' for p in predictions.values()],
                            'Win Probability': [f"{p:.2%}" for p in probabilities.values()],
                            'Confidence': [f"{max(p, 1-p)*100:.1f}%" for p in probabilities.values()]
                        })
                        
                        st.dataframe(model_results, use_container_width=True)
                        
                        # Visualization
                        if probabilities:
                            fig = px.bar(
                                x=list(probabilities.keys()), 
                                y=list(probabilities.values()),
                                title='🎯 Win Probability by Model',
                                labels={'x': 'Model', 'y': 'Win Probability'},
                                color=list(probabilities.values()),
                                color_continuous_scale='RdYlGn',
                                text=[f"{p:.2%}" for p in probabilities.values()]
                            )
                            fig.update_layout(showlegend=False, height=400)
                            fig.update_traces(textposition='outside')
                            fig.add_hline(y=0.5, line_dash="dash", line_color="black", 
                                         annotation_text="Decision Boundary (50%)")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        st.subheader("📊 Match Analysis")
                        
                        insights_col1, insights_col2 = st.columns(2)
                        
                        with insights_col1:
                            # Situation analysis
                            if rrr > 12:
                                situation = "🔥 Very Difficult"
                            elif rrr > 8:
                                situation = "⚠️ Challenging"
                            elif rrr > 6:
                                situation = "📈 Moderate"
                            else:
                                situation = "✅ Comfortable"
                            
                            st.info(f"""
                            **🎯 Match Situation**: {situation}
                            - Required Rate: {rrr:.2f} runs/over
                            - Current Rate: {crr:.2f} runs/over
                            - Wickets in Hand: {wickets}
                            - Balls Remaining: {balls_left}
                            """)
                        
                        with insights_col2:
                            # Team analysis
                            st.success(f"""
                            **🏏 Team Information**
                            - Batting: {batting_team}
                            - Bowling: {bowling_team}
                            - Venue: {city}
                            - Target: {total_runs} runs
                            - Runs Needed: {runs_left}
                            """)
                
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("Please check your inputs and try again.")
            
            # Quick prediction presets
            st.markdown("---")
            st.subheader("🚀 Quick Prediction Scenarios")
            
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            
            with preset_col1:
                if st.button("😰 Pressure Situation", key="preset1"):
                    st.info("Set: 45 runs needed, 18 balls, 3 wickets")
                    
            with preset_col2:
                if st.button("😌 Comfortable Chase", key="preset2"):
                    st.info("Set: 25 runs needed, 30 balls, 7 wickets")
                    
            with preset_col3:
                if st.button("🔥 Thriller Finish", key="preset3"):
                    st.info("Set: 15 runs needed, 12 balls, 2 wickets")
        
        else:
            st.error("❌ Unable to load data. Please check if the data file exists.")

# Model Comparison Page
elif page == "📈 Model Comparison":
    st.markdown('<h2 class="sub-header">📈 Detailed Model Comparison</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if st.session_state.models is None:
        st.warning("⚠️ Please train the models first by visiting the 'Model Training' page.")
        st.info("👆 Go to **Model Training** page and click '🚀 Start Model Training' to train the models.")
        
        # Show what's needed
        st.markdown("---")
        st.subheader("📋 What You Need")
        st.write("""
        To use the Model Comparison page, you need:
        1. **Trained Models**: Visit the Model Training page
        2. **Test Data**: Generated automatically during training
        3. **Evaluation Metrics**: Calculated after training
        """)
        
        # Create sample visualization for demo
        st.subheader("📊 Sample Model Comparison (Demo)")
        
        # Demo data
        demo_models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Decision Tree']
        demo_accuracy = [0.85, 0.83, 0.78, 0.75]
        demo_f1 = [0.84, 0.82, 0.77, 0.74]
        
        demo_df = pd.DataFrame({
            'Model': demo_models,
            'Accuracy': demo_accuracy,
            'F1-Score': demo_f1
        })
        
        fig = px.bar(demo_df, x='Model', y=['Accuracy', 'F1-Score'],
                    title='Sample Model Performance Comparison',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("👆 This is a sample visualization. Train models to see real results!")
    
    else:
        # Check if we have test data
        if st.session_state.X_test is None or st.session_state.y_test is None:
            st.warning("⚠️ Test data not available. Please retrain the models to generate test data.")
            st.info("👆 Go to **Model Training** page and retrain the models to generate test data.")
            
            # Show available models without test data
            st.subheader("🤖 Available Models (No Test Data)")
            model_list = list(st.session_state.models.keys())
            
            for i, model_name in enumerate(model_list, 1):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {model_name}**")
                with col2:
                    st.write("✅ Trained")
                with col3:
                    st.write("❌ No Test Data")
            
            st.markdown("---")
            st.info("💡 **Tip**: Retrain models to generate test data for comprehensive evaluation.")
        
        else:
            # Full model comparison interface
            st.success(f"✅ **Ready for Model Comparison!** {len(st.session_state.models)} models available with test data.")
            
            # Show basic model info
            st.subheader("📊 Available Models Overview")
            
            model_info_data = []
            for name in st.session_state.models.keys():
                cv_score = "N/A"
                if st.session_state.scores and name in st.session_state.scores:
                    cv_score = f"{st.session_state.scores[name]['mean_cv_score']:.4f}"
                
                model_info_data.append({
                    'Model': name,
                    'Status': '✅ Ready',
                    'CV Score': cv_score,
                    'Test Data': '✅ Available'
                })
            
            model_info_df = pd.DataFrame(model_info_data)
            st.dataframe(model_info_df, use_container_width=True)
            
            # Evaluate all models on test set
            st.markdown("---")
            if st.button("📊 Evaluate All Models on Test Set", type="primary"):
                with st.spinner("Evaluating models on test set..."):
                    
                    evaluation_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, (name, model) in enumerate(st.session_state.models.items()):
                        progress = (i + 1) / len(st.session_state.models)
                        progress_bar.progress(progress)
                        status_text.text(f"Evaluating {name}... ({i+1}/{len(st.session_state.models)})")
                        
                        try:
                            metrics, y_pred, y_pred_proba = evaluate_model(
                                model, st.session_state.X_test, st.session_state.y_test
                            )
                            if metrics is not None:
                                evaluation_results[name] = {
                                    'metrics': metrics,
                                    'predictions': y_pred,
                                    'probabilities': y_pred_proba
                                }
                        except Exception as e:
                            st.warning(f"⚠️ Error evaluating {name}: {str(e)}")
                            continue
                    
                    progress_bar.progress(1.0)
                    status_text.text("Evaluation completed!")
                    
                    if not evaluation_results:
                        st.error("❌ No models could be evaluated. Please check the data and retrain.")
                    else:
                        st.session_state.evaluation_results = evaluation_results
                        st.success("✅ Model evaluation completed!")
                        
                        # Create comprehensive comparison
                        st.markdown("---")
                        st.subheader("📊 Performance Metrics Comparison")
                        
                        # Metrics table
                        metrics_data = []
                        for name, results in evaluation_results.items():
                            metrics = results['metrics']
                            metrics_data.append({
                                'Model': name,
                                'Accuracy': f"{metrics['accuracy']:.4f}",
                                'Precision': f"{metrics['precision']:.4f}",
                                'Recall': f"{metrics['recall']:.4f}",
                                'F1-Score': f"{metrics['f1']:.4f}",
                                'AUC-ROC': f"{metrics['auc']:.4f}" if metrics['auc'] else 'N/A'
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Metrics visualization
                        metrics_for_viz = ['accuracy', 'precision', 'recall', 'f1']
                        viz_data = []
                        
                        for metric in metrics_for_viz:
                            for name, results in evaluation_results.items():
                                if results['metrics'][metric] is not None:
                                    viz_data.append({
                                        'Model': name,
                                        'Metric': metric.upper(),
                                        'Score': results['metrics'][metric]
                                    })
                        
                        if viz_data:
                            viz_df = pd.DataFrame(viz_data)
                            
                            fig = px.bar(viz_df, x='Model', y='Score', color='Metric',
                                        title='Model Performance Comparison',
                                        barmode='group')
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # ROC Curves
                        st.subheader("📈 ROC Curves Comparison")
                        
                        fig_roc = go.Figure()
                        roc_data_available = False
                        
                        for name, results in evaluation_results.items():
                            if results['probabilities'] is not None:
                                try:
                                    fpr, tpr, _ = roc_curve(st.session_state.y_test, results['probabilities'])
                                    auc_score = results['metrics']['auc']
                                    
                                    fig_roc.add_trace(go.Scatter(
                                        x=fpr, y=tpr,
                                        mode='lines',
                                        name=f'{name} (AUC = {auc_score:.3f})',
                                        line=dict(width=2)
                                    ))
                                    roc_data_available = True
                                except Exception as e:
                                    continue
                        
                        if roc_data_available:
                            # Add diagonal line
                            fig_roc.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Classifier',
                                line=dict(dash='dash', color='black')
                            ))
                            
                            fig_roc.update_layout(
                                title='ROC Curves Comparison',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                width=800, height=600
                            )
                            
                            st.plotly_chart(fig_roc, use_container_width=True)
                        else:
                            st.info("📋 ROC curves not available (models may not support probability prediction)")
                        
                        # Model Recommendations
                        st.subheader("🎯 Model Recommendations")
                        
                        if evaluation_results:
                            # Find best model based on different metrics
                            best_accuracy = max(evaluation_results.items(), key=lambda x: x[1]['metrics']['accuracy'])
                            best_f1 = max(evaluation_results.items(), key=lambda x: x[1]['metrics']['f1'])
                            
                            # Find best AUC (only from models that have AUC scores)
                            auc_models = [item for item in evaluation_results.items() if item[1]['metrics']['auc'] is not None]
                            best_auc = max(auc_models, key=lambda x: x[1]['metrics']['auc']) if auc_models else None
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>🎯 Best Accuracy</h4>
                                    <h3>{best_accuracy[0]}</h3>
                                    <p>{best_accuracy[1]['metrics']['accuracy']:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>⚖️ Best F1-Score</h4>
                                    <h3>{best_f1[0]}</h3>
                                    <p>{best_f1[1]['metrics']['f1']:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                if best_auc:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>📈 Best AUC-ROC</h4>
                                        <h3>{best_auc[0]}</h3>
                                        <p>{best_auc[1]['metrics']['auc']:.4f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="metric-card">
                                        <h4>📈 AUC-ROC</h4>
                                        <h3>N/A</h3>
                                        <p>Not Available</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Ensemble recommendation
                            st.markdown("---")
                            st.markdown("### 🏆 Recommended Approach")
                            st.info(f"""
                            **📈 Performance Analysis:**
                            
                            Based on the evaluation results:
                            - **Best Accuracy**: {best_accuracy[0]} ({best_accuracy[1]['metrics']['accuracy']:.4f})
                            - **Best F1-Score**: {best_f1[0]} ({best_f1[1]['metrics']['f1']:.4f})
                            - **Models Evaluated**: {len(evaluation_results)}
                            
                            **🎯 Recommendations:**
                            - Use **{best_f1[0]}** for balanced performance
                            - Consider ensemble of top 3 performing models
                            - Focus on F1-score for this classification task
                            """)
            
            # Show previous evaluation results if available
            elif st.session_state.evaluation_results is not None:
                st.info("📊 Previous evaluation results are available. Click the button above to re-evaluate or view the cached results below.")
                
                # Display cached results summary
                results = st.session_state.evaluation_results
                st.subheader("📈 Previous Evaluation Summary")
                
                summary_data = []
                for name, result in results.items():
                    metrics = result['metrics']
                    summary_data.append({
                        'Model': name,
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'F1-Score': f"{metrics['f1']:.4f}",
                        'Status': '✅ Evaluated'
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Quick visualization of cached results
                if len(summary_data) > 0:
                    accuracy_scores = [float(item['Accuracy']) for item in summary_data]
                    model_names = [item['Model'] for item in summary_data]
                    
                    fig = px.bar(x=model_names, y=accuracy_scores,
                               title='Previous Evaluation Results - Accuracy Comparison',
                               labels={'x': 'Model', 'y': 'Accuracy'},
                               color=accuracy_scores,
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("📊 Click the button above to evaluate all trained models on the test set.")
                
                # Show training summary
                st.subheader("📋 Training Summary")
                if st.session_state.scores:
                    training_summary = []
                    for name, score_data in st.session_state.scores.items():
                        training_summary.append({
                            'Model': name,
                            'CV Score': f"{score_data['mean_cv_score']:.4f}",
                            'CV Std': f"±{score_data['std_cv_score']:.4f}",
                            'Status': '✅ Trained'
                        })
                    
                    training_df = pd.DataFrame(training_summary)
                    st.dataframe(training_df, use_container_width=True)
                    
                    # Quick visualization
                    cv_scores = [score_data['mean_cv_score'] for score_data in st.session_state.scores.values()]
                    model_names = list(st.session_state.scores.keys())
                    
                    fig = px.bar(x=model_names, y=cv_scores,
                               title='Cross-Validation Scores (Training)',
                               labels={'x': 'Model', 'y': 'CV Score'},
                               color=cv_scores,
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No training scores available.")

# Save/Load Model functionality
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Model Management")

# Show model status in sidebar
if st.session_state.models is not None:
    st.sidebar.success(f"✅ {len(st.session_state.models)} models loaded")
    
    # Save models
    if st.sidebar.button("💾 Save Models"):
        try:
            model_data = {
                'models': st.session_state.models,
                'encoders': st.session_state.encoders,
                'scaler': st.session_state.scaler,
                'scores': st.session_state.scores
            }
            
            with open('cricket_models.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            st.sidebar.success("✅ Models saved successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error saving models: {e}")
else:
    st.sidebar.warning("⚠️ No models loaded")

# Load models
uploaded_file = st.sidebar.file_uploader("📁 Load Saved Models", type=['pkl'])
if uploaded_file is not None:
    try:
        model_data = pickle.load(uploaded_file)
        st.session_state.models = model_data.get('models')
        st.session_state.encoders = model_data.get('encoders')
        st.session_state.scaler = model_data.get('scaler')
        st.session_state.scores = model_data.get('scores', {})
        st.session_state.models_loaded = True
        st.sidebar.success("✅ Models loaded successfully!")
        st.rerun()  # Refresh the app
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")

# Reset models option
if st.session_state.models is not None:
    if st.sidebar.button("🔄 Reset Models", help="Clear all models and start fresh"):
        st.session_state.models = None
        st.session_state.encoders = None
        st.session_state.scaler = None
        st.session_state.scores = None
        st.session_state.models_loaded = False
        st.session_state.evaluation_results = None
        st.sidebar.success("✅ Models reset!")
        st.rerun()

# Model info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ App Info")
st.sidebar.info("""
**🎯 Quick Start:**
1. Models auto-load if available
2. Go to Prediction page to use
3. Train once if needed

**💡 Tip:** 
Models are saved automatically after training and loaded on app start.
""")

# Additional sidebar features
st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Status")

# Show current page status
if page == "🏠 Home":
    st.sidebar.info("🏠 Currently on Home page")
elif page == "📊 Data Analysis":
    st.sidebar.info("📊 Currently analyzing data")
elif page == "🤖 Model Training":
    if st.session_state.models is not None:
        st.sidebar.success("🤖 Models are trained")
    else:
        st.sidebar.warning("🤖 No models trained yet")
elif page == "🔮 Prediction":
    if st.session_state.models is not None:
        st.sidebar.success("🔮 Ready for predictions")
    else:
        st.sidebar.error("🔮 Models needed for prediction")
elif page == "📈 Model Comparison":
    if st.session_state.models is not None and st.session_state.X_test is not None:
        st.sidebar.success("📈 Ready for comparison")
    elif st.session_state.models is not None:
        st.sidebar.warning("📈 Models available, need test data")
    else:
        st.sidebar.error("📈 Models needed")

# Dataset info in sidebar
data = load_data()
if data is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Dataset Info")
    st.sidebar.metric("Total Records", f"{len(data):,}")
    st.sidebar.metric("Teams", len(data['batting_team'].unique()))
    st.sidebar.metric("Cities", len(data['city'].unique()))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🏏 Cricket Win Prediction System</h4>
    <p>Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
    <p>🚀 <strong>Ready-to-use prediction system</strong> - Train once, predict forever!</p>
    <p style="font-size: 0.8em; margin-top: 1rem;">
        💡 <strong>Usage Flow:</strong> Home → Data Analysis → Model Training → Prediction → Model Comparison
    </p>
</div>
""", unsafe_allow_html=True)