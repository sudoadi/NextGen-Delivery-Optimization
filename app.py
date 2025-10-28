import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="NexGen Logistics - Predictive Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configuration
def apply_theme():
    """Apply theme based on user selection"""
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        plt.style.use('dark_background')
        chart_bg = '#0e1117'
        text_color = 'white'
        grid_color = '#2b2b2b'
    else:
        plt.style.use('default')
        chart_bg = 'white'
        text_color = 'black'
        grid_color = '#f0f0f0'
    
    return theme, chart_bg, text_color, grid_color

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .risk-medium { 
        background: linear-gradient(135deg, #ffd93d 0%, #ff9a3d 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .risk-low { 
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .info-card {
        background: var(--secondary-background-color, #f8f9fa);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background: var(--secondary-background-color, #f8f9fa);
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Training and Loading Functions ---
def train_and_save_model():
    """Train the enhanced model with all features"""
    try:
        delivery_df = pd.read_csv("delivery_performance.csv")
        orders_df = pd.read_csv("orders.csv")
        routes_df = pd.read_csv("routes_distance.csv")
        cost_df = pd.read_csv("cost_breakdown.csv")
        inventory_df = pd.read_csv("warehouse_inventory.csv")
    except FileNotFoundError as e:
        st.error(f"Error: Could not find required data file: {e}")
        return None, None, None

    # Merge datasets
    data_df = pd.merge(delivery_df, orders_df, on="Order_ID", how="inner")
    data_df = pd.merge(data_df, routes_df, on="Order_ID", how="inner")
    data_df = pd.merge(data_df, cost_df, on="Order_ID", how="left")
    data_df = pd.merge(data_df, inventory_df, 
                       left_on=['Origin', 'Product_Category'], 
                       right_on=['Location', 'Product_Category'], 
                       how="left")

    # Feature Engineering
    data_df['is_delayed'] = data_df['Delivery_Status'].apply(lambda x: 0 if x == 'On-Time' else 1)
    data_df['Order_Date'] = pd.to_datetime(data_df['Order_Date'], errors='coerce')
    data_df['order_day_of_week'] = data_df['Order_Date'].dt.dayofweek
    data_df['distance_per_day'] = data_df['Distance_KM'] / (data_df['Promised_Delivery_Days'] + 1e-6)
    data_df['delay_days'] = data_df['Actual_Delivery_Days'] - data_df['Promised_Delivery_Days']
    data_df['stock_vs_reorder'] = data_df['Current_Stock_Units'] - data_df['Reorder_Level']
    data_df['stock_vs_reorder'].fillna(0, inplace=True)
    data_df['Cost_per_KM'] = data_df['Delivery_Cost_INR'] / data_df['Distance_KM']
    
    # Fill missing values
    categorical_features = ['Carrier', 'Customer_Segment', 'Priority', 'Product_Category', 
                           'Origin', 'Destination', 'Special_Handling', 'Weather_Impact']
    numerical_features = ['Promised_Delivery_Days', 'Order_Value_INR', 'Distance_KM', 
                         'Traffic_Delay_Minutes', 'order_day_of_week', 'distance_per_day',
                         'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance']

    for col in categorical_features:
        data_df[col] = data_df[col].fillna(data_df[col].mode()[0] if len(data_df[col].mode()) > 0 else "Unknown")
    for col in numerical_features:
        data_df[col] = data_df[col].fillna(data_df[col].median())

    # Prepare features and target
    X = data_df[categorical_features + numerical_features]
    y = data_df['is_delayed']

    # Create pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
    ])

    # Train model
    model.fit(X, y)
    
    # Save model
    model_filename = "enhanced_delivery_model.joblib"
    joblib.dump(model, model_filename)
    
    # Get feature importance
    try:
        ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([ohe_feature_names, numerical_features])
        importances = model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        top_features = feature_importance_df.head(10)['feature'].tolist()
    except:
        top_features = []

    st.success("Enhanced model trained and saved!")
    return model, top_features, data_df

@st.cache_resource
def load_model_and_data():
    """Load model and data with caching"""
    model_file = "enhanced_delivery_model.joblib"
    
    if not os.path.exists(model_file):
        st.info("Training enhanced predictive model...")
        model, top_features, data_df = train_and_save_model()
        if model is None:
            return None, None, None
    else:
        st.info("Loading pre-trained enhanced model...")
        model = joblib.load(model_file)
        
        # Load and process data
        try:
            delivery_df = pd.read_csv("delivery_performance.csv")
            orders_df = pd.read_csv("orders.csv")
            routes_df = pd.read_csv("routes_distance.csv")
            cost_df = pd.read_csv("cost_breakdown.csv")
            inventory_df = pd.read_csv("warehouse_inventory.csv")

            data_df = pd.merge(delivery_df, orders_df, on="Order_ID", how="inner")
            data_df = pd.merge(data_df, routes_df, on="Order_ID", how="inner")
            data_df = pd.merge(data_df, cost_df, on="Order_ID", how="left")
            data_df = pd.merge(data_df, inventory_df, 
                               left_on=['Origin', 'Product_Category'], 
                               right_on=['Location', 'Product_Category'], 
                               how="left")
            
            # Feature engineering
            data_df['is_delayed'] = data_df['Delivery_Status'].apply(lambda x: 0 if x == 'On-Time' else 1)
            data_df['Order_Date'] = pd.to_datetime(data_df['Order_Date'], errors='coerce')
            data_df['order_day_of_week'] = data_df['Order_Date'].dt.dayofweek
            data_df['distance_per_day'] = data_df['Distance_KM'] / (data_df['Promised_Delivery_Days'] + 1e-6)
            data_df['delay_days'] = data_df['Actual_Delivery_Days'] - data_df['Promised_Delivery_Days']
            data_df['stock_vs_reorder'] = data_df['Current_Stock_Units'] - data_df['Reorder_Level']
            data_df['stock_vs_reorder'].fillna(0, inplace=True)
            data_df['Cost_per_KM'] = data_df['Delivery_Cost_INR'] / data_df['Distance_KM']
            
            # Fill missing values
            categorical_features = ['Carrier', 'Customer_Segment', 'Priority', 'Product_Category', 
                                   'Origin', 'Destination', 'Special_Handling', 'Weather_Impact']
            numerical_features = ['Promised_Delivery_Days', 'Order_Value_INR', 'Distance_KM', 
                                 'Traffic_Delay_Minutes', 'order_day_of_week', 'distance_per_day',
                                 'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance']

            for col in categorical_features:
                data_df[col] = data_df[col].fillna(data_df[col].mode()[0] if len(data_df[col].mode()) > 0 else "Unknown")
            for col in numerical_features:
                data_df[col] = data_df[col].fillna(data_df[col].median())

            # Get feature importance
            try:
                ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
                all_feature_names = np.concatenate([ohe_feature_names, numerical_features])
                importances = model.named_steps['classifier'].feature_importances_
                feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                top_features = feature_importance_df.head(10)['feature'].tolist()
            except Exception as e:
                st.warning(f"Could not calculate feature importances: {e}")
                top_features = []

        except FileNotFoundError:
            st.error("Could not find data files. Please ensure all required CSVs are present.")
            return None, None, None
    
    # Calculate probabilities for all orders
    st.info("Calculating risk probabilities for all orders...")
    try:
        categorical_features = ['Carrier', 'Customer_Segment', 'Priority', 'Product_Category', 
                               'Origin', 'Destination', 'Special_Handling', 'Weather_Impact']
        numerical_features = ['Promised_Delivery_Days', 'Order_Value_INR', 'Distance_KM', 
                             'Traffic_Delay_Minutes', 'order_day_of_week', 'distance_per_day',
                             'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance']
        X_data = data_df[categorical_features + numerical_features]
        probabilities = model.predict_proba(X_data)[:, 1]
        data_df['predicted_probability'] = probabilities
    except Exception as e:
        st.error(f"Error running predictions: {e}")
        data_df['predicted_probability'] = 0.0

    st.success("Enhanced model and data loaded successfully!")
    return model, data_df, top_features

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "HIGH", "🔴", "Immediate action required"
    elif probability >= 0.4:
        return "MEDIUM", "🟡", "Close monitoring needed"
    else:
        return "LOW", "🟢", "Standard procedures"

def create_dynamic_charts(df, selected_order_data, theme, chart_bg, text_color, grid_color):
    """Create dynamic charts that update based on selections"""
    charts = {}
    
    # Risk Distribution Chart
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    risk_bins = [0, 0.3, 0.7, 1.0]
    risk_labels = ['Low', 'Medium', 'High']
    df['Risk_Level'] = pd.cut(df['predicted_probability'], bins=risk_bins, labels=risk_labels)
    risk_counts = df['Risk_Level'].value_counts()
    
    colors = ['#51cf66', '#ffa94d', '#ff6b6b']
    bars = ax1.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.8)
    ax1.set_title('Order Risk Distribution', color=text_color, fontweight='bold')
    ax1.set_ylabel('Number of Orders', color=text_color)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', color=text_color, fontweight='bold')
    
    ax1.set_facecolor(chart_bg)
    fig1.patch.set_facecolor(chart_bg)
    ax1.tick_params(colors=text_color)
    ax1.grid(True, alpha=0.3, color=grid_color)
    
    charts['risk_distribution'] = fig1
    
    # Carrier Performance Comparison
    carrier_stats = df.groupby('Carrier').agg({
        'is_delayed': 'mean',
        'Delivery_Cost_INR': 'mean',
        'Order_ID': 'count'
    }).round(3)
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    x = range(len(carrier_stats))
    width = 0.35
    
    bars1 = ax2.bar([i - width/2 for i in x], carrier_stats['is_delayed'] * 100, 
                   width, label='Delay Rate %', color='#ff6b6b', alpha=0.7)
    bars2 = ax2.bar([i + width/2 for i in x], carrier_stats['Delivery_Cost_INR'], 
                   width, label='Avg Cost (INR)', color='#1f77b4', alpha=0.7)
    
    ax2.set_xlabel('Carrier', color=text_color)
    ax2.set_ylabel('Metrics', color=text_color)
    ax2.set_title('Carrier Performance: Delay Rate vs Cost', color=text_color, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(carrier_stats.index, rotation=45, ha='right', color=text_color)
    ax2.legend(facecolor=chart_bg, edgecolor=text_color)
    
    ax2.set_facecolor(chart_bg)
    fig2.patch.set_facecolor(chart_bg)
    ax2.tick_params(colors=text_color)
    ax2.grid(True, alpha=0.3, color=grid_color)
    
    charts['carrier_performance'] = fig2
    
    return charts

def get_business_insights(data_df):
    """Generate comprehensive business insights"""
    insights = {}
    
    # Cost Analysis
    cost_delayed = data_df[data_df['is_delayed'] == 1]['Delivery_Cost_INR'].mean()
    cost_ontime = data_df[data_df['is_delayed'] == 0]['Delivery_Cost_INR'].mean()
    insights['cost_analysis'] = {
        'delayed_avg_cost': cost_delayed,
        'ontime_avg_cost': cost_ontime,
        'cost_difference': cost_delayed - cost_ontime
    }
    
    # Carrier Performance
    carrier_stats = data_df.groupby('Carrier').agg({
        'is_delayed': 'mean',
        'Delivery_Cost_INR': 'mean',
        'Customer_Rating': 'mean',
        'Order_ID': 'count'
    }).round(3)
    insights['carrier_performance'] = carrier_stats
    
    # Route Analysis
    route_stats = data_df.groupby('Route').agg({
        'predicted_probability': 'mean',
        'is_delayed': 'mean',
        'Delivery_Cost_INR': 'mean'
    }).round(3)
    insights['route_analysis'] = route_stats
    
    return insights

def main():
    """Main application function"""
    
    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'selected_order' not in st.session_state:
        st.session_state.selected_order = None
    
    # Apply theme
    theme, chart_bg, text_color, grid_color = apply_theme()
    
    # Header with theme toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header">🚚 NexGen Logistics Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Predictive Delivery & Business Analytics Platform</div>', unsafe_allow_html=True)
    with col3:
        theme_option = st.selectbox("Theme", ["light", "dark"], index=0 if theme == "light" else 1, 
                                  key='theme_selector', on_change=lambda: st.session_state.update(theme=st.session_state.theme_selector))
    
    # Load model and data
    model, data_df, top_features = load_model_and_data()
    
    if model is None or data_df is None:
        st.error("Application cannot start. Please check data files and try again.")
        st.stop()
    
    # Generate business insights
    business_insights = get_business_insights(data_df)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Business Intelligence", "🔍 Order Analytics", "🎯 Predictive Dashboard"])
    
    with tab1:
        st.header("Business Intelligence & Cost Analytics")
        
        # KPI Section
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_orders = len(data_df)
            st.metric("Total Orders", f"{total_orders:,}")
        
        with col2:
            delay_rate = data_df['is_delayed'].mean() * 100
            st.metric("Overall Delay Rate", f"{delay_rate:.1f}%")
        
        with col3:
            cost_diff = business_insights['cost_analysis']['cost_difference']
            st.metric("Cost Impact of Delays", f"₹{cost_diff:.2f}", "per delivery", delta_color="inverse")
        
        with col4:
            high_risk_orders = len(data_df[data_df['predicted_probability'] >= 0.7])
            st.metric("High Risk Orders", f"{high_risk_orders:,}")
        
        st.divider()
        
        # Cost Analysis
        st.subheader("Cost & Efficiency Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost breakdown chart using Altair
            cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost']
            cost_totals = data_df[cost_columns].sum().reset_index()
            cost_totals.columns = ['Cost_Type', 'Total_Cost']
            
            chart = alt.Chart(cost_totals).mark_bar().encode(
                x=alt.X('Total_Cost', title='Total Cost (INR)'),
                y=alt.Y('Cost_Type', sort='-x', title='Cost Category'),
                color=alt.Color('Cost_Type', legend=None),
                tooltip=['Cost_Type', 'Total_Cost']
            ).properties(title='Operational Cost Breakdown')
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            # Carrier cost efficiency
            carrier_efficiency = data_df.groupby('Carrier').agg({
                'Delivery_Cost_INR': 'mean',
                'is_delayed': 'mean',
                'Distance_KM': 'mean'
            }).round(2)
            carrier_efficiency['Cost_per_KM'] = carrier_efficiency['Delivery_Cost_INR'] / carrier_efficiency['Distance_KM']
            
            st.dataframe(
                carrier_efficiency.sort_values('Cost_per_KM'),
                use_container_width=True,
                column_config={
                    "Delivery_Cost_INR": "Avg Cost",
                    "is_delayed": "Delay Rate",
                    "Distance_KM": "Avg Distance",
                    "Cost_per_KM": "Cost/KM"
                }
            )
        
        st.divider()
        
        # Operational Deep Dive
        st.header("Operational Optimization")
        
        opt_tab1, opt_tab2, opt_tab3 = st.tabs(["🚚 Carrier Performance", "🗺️ Route Analytics", "📈 Risk Analysis"])
        
        with opt_tab1:
            carrier_stats = business_insights['carrier_performance']
            best_carrier = carrier_stats.sort_values(by=['is_delayed', 'Delivery_Cost_INR']).iloc[0]
            worst_carrier = carrier_stats.sort_values(by=['is_delayed', 'Delivery_Cost_INR'], ascending=[False, False]).iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Performer", 
                         f"{best_carrier.name}",
                         f"Delay: {best_carrier['is_delayed']:.1%} | Cost: ₹{best_carrier['Delivery_Cost_INR']:.0f}")
            
            with col2:
                st.metric("Worst Performer", 
                         f"{worst_carrier.name}",
                         f"Delay: {worst_carrier['is_delayed']:.1%} | Cost: ₹{worst_carrier['Delivery_Cost_INR']:.0f}",
                         delta_color="inverse")
            
            st.dataframe(carrier_stats, use_container_width=True)
        
        with opt_tab2:
            route_analysis = business_insights['route_analysis']
            st.dataframe(
                route_analysis.sort_values('predicted_probability', ascending=False).head(10),
                use_container_width=True,
                column_config={
                    "predicted_probability": "Risk Score",
                    "is_delayed": "Actual Delay Rate",
                    "Delivery_Cost_INR": "Avg Cost"
                }
            )
        
        with opt_tab3:
            st.subheader("High-Risk Order Management")
            high_risk_data = data_df[data_df['predicted_probability'] >= 0.7][[
                'Order_ID', 'predicted_probability', 'Carrier', 'Route', 'Delivery_Cost_INR'
            ]].sort_values('predicted_probability', ascending=False)
            
            st.dataframe(high_risk_data.head(10), use_container_width=True)
    
    with tab2:
        st.header("Individual Order Analytics")
        
        # Order selection in sidebar
        st.sidebar.header("Order Analysis")
        order_list = [""] + data_df['Order_ID'].unique().tolist()
        selected_order = st.sidebar.selectbox("Select Order:", order_list)
        
        if selected_order:
            order_data = data_df[data_df['Order_ID'] == selected_order].iloc[0]
            
            # Order details
            st.subheader(f"Order Analysis: {selected_order}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Order Details")
                st.markdown(f'<div class="info-card">', unsafe_allow_html=True)
                st.write(f"**Route:** {order_data['Origin']} → {order_data['Destination']}")
                st.write(f"**Carrier:** {order_data['Carrier']}")
                st.write(f"**Priority:** {order_data['Priority']}")
                st.write(f"**Product:** {order_data['Product_Category']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Delivery Metrics")
                st.markdown(f'<div class="info-card">', unsafe_allow_html=True)
                st.write(f"**Distance:** {order_data['Distance_KM']:,.0f} KM")
                st.write(f"**Promised Days:** {order_data['Promised_Delivery_Days']}")
                st.write(f"**Actual Days:** {order_data['Actual_Delivery_Days']}")
                st.write(f"**Cost:** ₹{order_data['Delivery_Cost_INR']:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk prediction
            st.markdown("### 🎯 Risk Assessment")
            probability = order_data['predicted_probability']
            risk_level, risk_emoji, risk_description = get_risk_level(probability)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if risk_level == "HIGH":
                    st.markdown(f'<div class="risk-high"><h4>{risk_emoji} HIGH RISK</h4><p>{risk_description}</p></div>', unsafe_allow_html=True)
                elif risk_level == "MEDIUM":
                    st.markdown(f'<div class="risk-medium"><h4>{risk_emoji} MEDIUM RISK</h4><p>{risk_description}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low"><h4>{risk_emoji} LOW RISK</h4><p>{risk_description}</p></div>', unsafe_allow_html=True)
                
                st.metric("Delay Probability", f"{probability:.1%}")
                st.progress(probability)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if risk_level == "HIGH":
                recommendations = [
                    "🚨 Contact customer about potential delay",
                    "⚡ Consider expedited shipping options",
                    "👨‍💼 Assign to experienced driver",
                    "⏰ Add 2-3 day buffer to delivery promise",
                    "📞 Monitor every 2 hours"
                ]
            elif risk_level == "MEDIUM":
                recommendations = [
                    "⚠️ Optimize route to avoid traffic",
                    "📦 Ensure proper packaging",
                    "🔍 Monitor progress twice daily",
                    "💬 Prepare delay notification",
                    "🔄 Confirm carrier availability"
                ]
            else:
                recommendations = [
                    "✅ Continue standard monitoring",
                    "📊 Maintain current procedures",
                    "🔄 Regular quality checks",
                    "📋 Standard communication"
                ]
            
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("Predictive Risk Dashboard")
        
        # Dynamic charts
        selected_order_data = data_df[data_df['Order_ID'] == st.session_state.selected_order].iloc[0] if st.session_state.selected_order else None
        charts = create_dynamic_charts(data_df, selected_order_data, theme, chart_bg, text_color, grid_color)
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(charts['risk_distribution'])
        with col2:
            st.pyplot(charts['carrier_performance'])
        
        # Risk heatmaps
        st.subheader("Risk Heatmaps")
        col1, col2 = st.columns(2)
        
        with col1:
            # Carrier vs Origin heatmap
            heatmap_data = data_df.groupby(['Carrier', 'Origin'])['predicted_probability'].mean().reset_index()
            chart = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('Origin', title='Origin'),
                y=alt.Y('Carrier', title='Carrier'),
                color=alt.Color('predicted_probability', title='Risk', scale=alt.Scale(scheme='redyellowgreen')),
                tooltip=['Carrier', 'Origin', alt.Tooltip('predicted_probability', format='.1%')]
            ).properties(title='Risk: Carrier vs Origin')
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            # Day of week analysis
            day_mapping = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            data_df['day_name'] = data_df['order_day_of_week'].map(day_mapping)
            day_risk = data_df.groupby('day_name')['predicted_probability'].mean().reset_index()
            
            chart = alt.Chart(day_risk).mark_bar().encode(
                x=alt.X('day_name', title='Day of Week'),
                y=alt.Y('predicted_probability', title='Average Risk', axis=alt.Axis(format='%')),
                color=alt.Color('predicted_probability', scale=alt.Scale(scheme='redyellowgreen'), legend=None),
                tooltip=['day_name', alt.Tooltip('predicted_probability', format='.1%')]
            ).properties(title='Risk by Day of Week')
            st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()