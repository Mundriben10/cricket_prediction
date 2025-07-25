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

# Sidebar for navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Choose a page:", 
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Prediction", "📈 Model Comparison"])

@st.cache_data
def load_data():
    """Load and preprocess the cricket data"""
    try:
        data = pd.read_csv('final_data.csv')
        # Drop the unnamed index column
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        if '' in data.columns:
            data = data.drop('', axis=1)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for machine learning"""
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
    
    X = df_processed[feature_cols]
    y = df_processed['result']
    
    return X, y, le_batting, le_bowling, le_city

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
    
    return trained_models, model_scores

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }
    
    return metrics, y_pred, y_pred_proba

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
                city_wins = city_wins[city_wins['count'] >= 100]  # Filter cities with at least 100 matches
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
                bins = np.arange(0, data['runs_left'].max() + 20, 20)
                data['runs_left_bin'] = pd.cut(data['runs_left'], bins)
                runs_analysis = data.groupby('runs_left_bin')['result'].agg(['mean', 'count']).reset_index()
                runs_analysis = runs_analysis[runs_analysis['count'] >= 50]
                runs_analysis['runs_left_mid'] = runs_analysis['runs_left_bin'].apply(lambda x: x.mid)
                
                fig = px.line(runs_analysis, x='runs_left_mid', y='mean',
                            title='Win Probability vs Runs Remaining',
                            labels={'runs_left_mid': 'Runs Remaining', 'mean': 'Win Probability'})
                fig.update_traces(line_color='red', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Required run rate vs Win probability
                data['rrr_bin'] = pd.cut(data['rrr'], bins=np.arange(0, 20, 1))
                rrr_analysis = data.groupby('rrr_bin')['result'].agg(['mean', 'count']).reset_index()
                rrr_analysis = rrr_analysis[rrr_analysis['count'] >= 100]
                rrr_analysis['rrr_mid'] = rrr_analysis['rrr_bin'].apply(lambda x: x.mid)
                
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
        
        if st.button("🚀 Start Model Training", key="train_button"):
            with st.spinner("Preparing data and initializing models..."):
                # Preprocess data
                X, y, le_batting, le_bowling, le_city = preprocess_data(data)
                
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
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
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
            
            # Display comprehensive training results
            st.markdown("---")
            st.subheader("📊 Training Results Summary")
            
            # Create two columns for results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Detailed scores table
                scores_df = pd.DataFrame(st.session_state.scores).T
                scores_df['mean_cv_score'] = scores_df['mean_cv_score'].round(4)
                scores_df['std_cv_score'] = scores_df['std_cv_score'].round(4)
                scores_df = scores_df.sort_values('mean_cv_score', ascending=False)
                scores_df['Rank'] = range(1, len(scores_df) + 1)
                
                # Add performance categories
                scores_df['Performance'] = scores_df['mean_cv_score'].apply(
                    lambda x: '🥇 Excellent' if x >= 0.85 else
                              '🥈 Good' if x >= 0.80 else
                              '🥉 Fair' if x >= 0.75 else
                              '📉 Needs Improvement'
                )
                
                # Reorder columns
                scores_df = scores_df[['Rank', 'mean_cv_score', 'std_cv_score', 'Performance']]
                scores_df.columns = ['Rank', 'CV Score', 'Std Dev', 'Performance']
                
                st.dataframe(scores_df, use_container_width=True)
            
            with col2:
                # Best model highlight
                best_model = scores_df.index[0]
                best_score = scores_df.iloc[0]['CV Score']
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>🏆 Champion Model</h4>
                    <h3>{best_model}</h3>
                    <h4>Score: {best_score:.4f}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive performance visualization
            fig = px.bar(
                scores_df.reset_index(), 
                x='index', 
                y='CV Score',
                error_y='Std Dev',
                title='🎯 Model Performance Comparison (Cross-Validation)',
                labels={'index': 'Model', 'CV Score': 'Accuracy Score'},
                color='CV Score',
                color_continuous_scale='RdYlGn',
                text='CV Score'
            )
            
            # Customize the chart
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(
                height=500,
                xaxis_tickangle=-45,
                showlegend=False
            )
            
            # Add a horizontal line for average performance
            avg_score = scores_df['CV Score'].mean()
            fig.add_hline(
                y=avg_score, 
                line_dash="dash", 
                line_color="orange",
                annotation_text=f"Average: {avg_score:.3f}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison insights
            st.subheader("💡 Training Insights")
            
            # Calculate insights
            score_range = scores_df['CV Score'].max() - scores_df['CV Score'].min()
            top_3_models = scores_df.head(3).index.tolist()
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.info(f"""
                **📈 Performance Analysis:**
                - Best performing model: **{best_model}** ({best_score:.4f})
                - Performance range: {score_range:.4f}
                - Models above average: {len(scores_df[scores_df['CV Score'] > avg_score])}
                """)
            
            with insights_col2:
                st.success(f"""
                **🎯 Recommendations:**
                - Top 3 models: {', '.join(top_3_models)}
                - Consider ensemble of top performers
                - All models ready for evaluation on test set
                """)
            
            # Training completion metrics
            st.markdown("---")
            st.balloons()  # Celebration animation
            
            completion_col1, completion_col2, completion_col3 = st.columns(3)
            with completion_col1:
                st.metric("✅ Models Trained", f"{len(models)}/6")
            with completion_col2:
                st.metric("🎯 Best CV Score", f"{best_score:.4f}")
            with completion_col3:
                st.metric("📊 Ready for Testing", "Yes")
        
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
            """)
    
    else:
        st.error("❌ Unable to load the dataset. Please ensure 'final_data.csv' is in the correct location.")

# Prediction Page
elif page == "🔮 Prediction":
    st.markdown('<h2 class="sub-header">🔮 Match Win Prediction</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is not None and 'models' in st.session_state:
        
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
        
        if st.button("🎯 Predict Win Probability", key="predict_button"):
            # Prepare input data
            le_batting, le_bowling, le_city = st.session_state.encoders
            
            try:
                batting_encoded = le_batting.transform([batting_team])[0]
                bowling_encoded = le_bowling.transform([bowling_team])[0]
                city_encoded = le_city.transform([city])[0]
                
                input_data = np.array([[batting_encoded, bowling_encoded, city_encoded, 
                                      runs_left, balls_left, wickets, total_runs, crr, rrr]])
                
                # Get predictions from all models
                predictions = {}
                probabilities = {}
                
                for name, model in st.session_state.models.items():
                    pred = model.predict(input_data)[0]
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(input_data)[0][1]  # Probability of winning
                    else:
                        prob = pred  # For models without predict_proba
                    
                    predictions[name] = pred
                    probabilities[name] = prob
                
                # Ensemble prediction (voting)
                ensemble_prob = np.mean(list(probabilities.values()))
                ensemble_pred = 1 if ensemble_prob > 0.5 else 0
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Main prediction card
                result_text = "WIN" if ensemble_pred == 1 else "LOSE"
                confidence = max(ensemble_prob, 1 - ensemble_prob) * 100
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🏆 Predicted Outcome: {result_text}</h2>
                    <h3>🎯 Win Probability: {ensemble_prob:.2%}</h3>
                    <h4>📊 Confidence Level: {confidence:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model predictions
                st.subheader("🤖 Individual Model Predictions")
                
                model_results = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Prediction': ['WIN' if p == 1 else 'LOSE' for p in predictions.values()],
                    'Win Probability': [f"{p:.2%}" for p in probabilities.values()]
                })
                
                st.dataframe(model_results, use_container_width=True)
                
                # Visualization
                fig = px.bar(x=list(probabilities.keys()), y=list(probabilities.values()),
                           title='Win Probability by Model',
                           labels={'x': 'Model', 'y': 'Win Probability'},
                           color=list(probabilities.values()),
                           color_continuous_scale='RdYlGn')
                fig.update_layout(showlegend=False)
                fig.add_hline(y=0.5, line_dash="dash", line_color="black", 
                             annotation_text="Decision Boundary")
                st.plotly_chart(fig, use_container_width=True)
                
            except ValueError as e:
                st.error(f"Error: {e}. Please check if the selected team/city exists in the training data.")
    
    else:
        st.warning("⚠️ Please train the models first by visiting the 'Model Training' page.")

# Model Comparison Page
elif page == "📈 Model Comparison":
    st.markdown('<h2 class="sub-header">📈 Detailed Model Comparison</h2>', unsafe_allow_html=True)
    
    if 'models' in st.session_state:
        # Evaluate all models on test set
        if st.button("📊 Evaluate Models on Test Set"):
            with st.spinner("Evaluating models on test set..."):
                
                evaluation_results = {}
                
                for name, model in st.session_state.models.items():
                    metrics, y_pred, y_pred_proba = evaluate_model(
                        model, st.session_state.X_test, st.session_state.y_test
                    )
                    evaluation_results[name] = {
                        'metrics': metrics,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                
                st.session_state.evaluation_results = evaluation_results
            
            st.success("✅ Model evaluation completed!")
            
            # Create comprehensive comparison
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
            
            viz_df = pd.DataFrame(viz_data)
            
            fig = px.bar(viz_df, x='Model', y='Score', color='Metric',
                        title='Model Performance Comparison',
                        barmode='group')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curves
            st.subheader("📈 ROC Curves Comparison")
            
            fig_roc = go.Figure()
            
            for name, results in evaluation_results.items():
                if results['probabilities'] is not None:
                    fpr, tpr, _ = roc_curve(st.session_state.y_test, results['probabilities'])
                    auc_score = results['metrics']['auc']
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{name} (AUC = {auc_score:.3f})',
                        line=dict(width=2)
                    ))
            
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
            
            # Confusion Matrices
            st.subheader("🔍 Confusion Matrices")
            
            # Create subplots for confusion matrices
            n_models = len(evaluation_results)
            cols = 3
            rows = (n_models + cols - 1) // cols
            
            fig_cm = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=list(evaluation_results.keys()),
                specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)]
            )
            
            for i, (name, results) in enumerate(evaluation_results.items()):
                row = i // cols + 1
                col = i % cols + 1
                
                cm = confusion_matrix(st.session_state.y_test, results['predictions'])
                
                fig_cm.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['Predicted 0', 'Predicted 1'],
                        y=['Actual 0', 'Actual 1'],
                        colorscale='Blues',
                        showscale=False,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16}
                    ),
                    row=row, col=col
                )
            
            fig_cm.update_layout(height=200*rows, title_text="Confusion Matrices")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            st.subheader("🌟 Feature Importance Analysis")
            
            feature_names = ['Batting Team', 'Bowling Team', 'City', 'Runs Left', 
                           'Balls Left', 'Wickets', 'Total Runs', 'Current RR', 'Required RR']
            
            importance_data = []
            
            for name, model in st.session_state.models.items():
                if hasattr(model, 'feature_importances_'):
                    for i, importance in enumerate(model.feature_importances_):
                        importance_data.append({
                            'Model': name,
                            'Feature': feature_names[i],
                            'Importance': importance
                        })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                
                fig_imp = px.bar(importance_df, x='Feature', y='Importance', color='Model',
                               title='Feature Importance Comparison',
                               barmode='group')
                fig_imp.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # Model Recommendations
            st.subheader("🎯 Model Recommendations")
            
            # Find best model based on different metrics
            best_accuracy = max(evaluation_results.items(), key=lambda x: x[1]['metrics']['accuracy'])
            best_f1 = max(evaluation_results.items(), key=lambda x: x[1]['metrics']['f1'])
            best_auc = max([item for item in evaluation_results.items() if item[1]['metrics']['auc'] is not None], 
                          key=lambda x: x[1]['metrics']['auc'])
            
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
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📈 Best AUC-ROC</h4>
                    <h3>{best_auc[0]}</h3>
                    <p>{best_auc[1]['metrics']['auc']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Ensemble recommendation
            st.markdown("---")
            st.markdown("### 🏆 Recommended Approach")
            st.info("""
            **Ensemble Method Recommendation:**
            
            Based on the evaluation results, consider using an ensemble of the top 3 performing models:
            1. Combine predictions from Random Forest, Gradient Boosting, and Logistic Regression
            2. Use weighted voting based on individual model performance
            3. This approach typically provides better generalization and robustness
            
            **For Production Use:**
            - Use the model with highest F1-score for balanced performance
            - Consider the model with highest AUC-ROC if probability calibration is important
            - Random Forest often provides good interpretability and performance balance
            """)
    
    else:
        st.warning("⚠️ Please train the models first by visiting the 'Model Training' page.")

# Save/Load Model functionality
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Model Management")

if 'models' in st.session_state:
    # Save models
    if st.sidebar.button("💾 Save Models"):
        try:
            model_data = {
                'models': st.session_state.models,
                'encoders': st.session_state.encoders,
                'scaler': st.session_state.scaler
            }
            
            with open('cricket_models.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            st.sidebar.success("✅ Models saved successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error saving models: {e}")

# Load models
uploaded_file = st.sidebar.file_uploader("📁 Load Saved Models", type=['pkl'])
if uploaded_file is not None:
    try:
        model_data = pickle.load(uploaded_file)
        st.session_state.models = model_data['models']
        st.session_state.encoders = model_data['encoders']
        st.session_state.scaler = model_data['scaler']
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🏏 Cricket Win Prediction System</h4>
    <p>Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
    <p>Predict cricket match outcomes with advanced machine learning!</p>
</div>
""", unsafe_allow_html=True)