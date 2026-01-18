"""
===============================================================================
Enhanced Railway Customer Care Streamlit App
Graph RAG + LoRA Fine-tuned Customer Support
===============================================================================

Installation:
pip install streamlit pandas numpy networkx plotly

Run:
streamlit run streamlit_customer_care_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Import the railway system
sys.path.append(os.path.dirname(__file__))
from railway_graph_rag_system import GraphRAGBot

# Page config
st.set_page_config(
    page_title="🚂 Indian Railways Customer Care",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #C70039;
        color: white;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    .delay-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #FF5733;
        font-weight: 700;
    }
    h2 {
        color: #C70039;
        font-weight: 600;
    }
    h3 {
        color: #900C3F;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_railway_system():
    """Load the railway system once"""
    bot = GraphRAGBot('data.csv')
    bot.initialize()
    return bot


def format_chat_message(role, content):
    """Format chat message with styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "🧑" if role == "user" else "🤖"
    return f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.upper()}:</strong><br/>
        {content}
    </div>
    """


def create_delay_visualization(scenarios):
    """Create delay probability visualization"""
    probs = [int(s[0]*100) for s in scenarios]
    delays = [int(s[1]) for s in scenarios]
    labels = [s[2] for s in scenarios]
    
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=delays,
            text=[f"{d} min<br>{p}%" for d, p in zip(delays, probs)],
            textposition='auto',
            marker=dict(
                color=colors[:len(delays)],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>Delay: %{y} min<br><extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Delay Prediction Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Expected Delay (minutes)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_weather_impact_chart(bot):
    """Create weather impact visualization"""
    weather_data = bot.analyzer.df.groupby('WeatherCondition')['Arrival_Delay_min'].agg(['mean', 'count'])
    weather_data = weather_data[weather_data['count'] > 50].sort_values('mean', ascending=False).head(6)
    
    fig = go.Figure(data=[
        go.Bar(
            x=weather_data.index,
            y=weather_data['mean'],
            marker=dict(
                color=weather_data['mean'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Avg Delay (min)")
            ),
            text=[f"{v:.1f} min" for v in weather_data['mean']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Weather Conditions Impact on Delays",
        xaxis_title="Weather Condition",
        yaxis_title="Average Delay (minutes)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_hourly_pattern_chart(bot):
    """Create hourly delay pattern"""
    hourly = bot.analyzer.df.groupby('Departure_Hour')['Arrival_Delay_min'].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly.index,
        y=hourly.values,
        mode='lines+markers',
        line=dict(color='#FF5733', width=3),
        marker=dict(size=8, color='#C70039'),
        fill='tozeroy',
        fillcolor='rgba(255, 87, 51, 0.2)',
        name='Avg Delay'
    ))
    
    fig.update_layout(
        title="Delay Patterns Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Delay (minutes)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    return fig


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("""
        <h1 style='text-align: center;'>🚂 Indian Railways Customer Care</h1>
        <p style='text-align: center; font-size: 18px; color: #666;'>
            AI-Powered Delay Prediction & Real-time Assistance
        </p>
    """, unsafe_allow_html=True)
    
    # Load system
    with st.spinner("🔄 Initializing Railway System..."):
        bot = load_railway_system()
    
    # Sidebar
    with st.sidebar:
        st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://irifm.indianrailways.gov.in/website/custom/images/indian_railways_logo.png" width="100">
    </div>
    """,
    unsafe_allow_html=True
)
        st.markdown("### 🎯 Quick Actions")
        
        page = st.radio(
            "Select Service:",
            ["🏠 Home", "💬 Customer Care Chat", "📊 Network Analytics", "⚙️ Delay Predictor"]
        )
        
        st.markdown("---")
        st.markdown("### 📞 Emergency Contact")
        st.info("**Helpline:** 139\n\n**SMS:** Your PNR to 139")
        
        st.markdown("---")
        st.markdown("### 🌟 Features")
        st.markdown("""
        - ✅ Real-time delay prediction
        - ✅ Weather impact analysis
        - ✅ Cascading delay detection
        - ✅ 24/7 AI assistance
        - ✅ Historical pattern insights
        """)
    
    # Main content
    if page == "🏠 Home":
        show_home_page(bot)
    elif page == "💬 Customer Care Chat":
        show_chat_page(bot)
    elif page == "📊 Network Analytics":
        show_analytics_page(bot)
    elif page == "⚙️ Delay Predictor":
        show_predictor_page(bot)


def show_home_page(bot):
    """Home dashboard"""
    st.markdown("## 🏠 Railway Network Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #4CAF50; margin: 0;'>📊 Total Trains</h3>
            <h1 style='margin: 5px 0;'>{:,}</h1>
            <p style='color: #666; margin: 0;'>Active Services</p>
        </div>
        """.format(len(bot.df)), unsafe_allow_html=True)
    
    with col2:
        avg_delay = bot.df['Arrival_Delay_min'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #FF9800; margin: 0;'>⏱️ Avg Delay</h3>
            <h1 style='margin: 5px 0;'>{:.1f} min</h1>
            <p style='color: #666; margin: 0;'>Network Average</p>
        </div>
        """.format(avg_delay), unsafe_allow_html=True)
    
    with col3:
        ontime_pct = (bot.df['Arrival_Delay_min'] <= 5).sum() / len(bot.df) * 100
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #2196F3; margin: 0;'>✅ On-Time %</h3>
            <h1 style='margin: 5px 0;'>{:.1f}%</h1>
            <p style='color: #666; margin: 0;'>Punctuality Rate</p>
        </div>
        """.format(ontime_pct), unsafe_allow_html=True)
    
    with col4:
        fog_penalty = bot.knowledge['fog']['penalty']
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #9C27B0; margin: 0;'>🌫️ Fog Impact</h3>
            <h1 style='margin: 5px 0;'>+{:.1f} min</h1>
            <p style='color: #666; margin: 0;'>Additional Delay</p>
        </div>
        """.format(fog_penalty), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_weather_impact_chart(bot), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_hourly_pattern_chart(bot), use_container_width=True)
    
    # Recent alerts
    st.markdown("## 🚨 Service Alerts")
    
    late_trains = bot.df[bot.df['Arrival_Delay_min'] > 60].head(5)
    
    if len(late_trains) > 0:
        for idx, row in late_trains.iterrows():
            delay = int(row['Arrival_Delay_min'])
            weather = row['WeatherCondition']
            
            if delay > 120:
                alert_type = "error"
                icon = "🔴"
            elif delay > 60:
                alert_type = "warning"
                icon = "🟠"
            else:
                alert_type = "info"
                icon = "🟡"
            
            with st.expander(f"{icon} Train #{idx} - {delay} minutes delay ({weather})"):
                st.write(f"**Distance:** {row['Distance_km']} km")
                st.write(f"**Scheduled Time:** {row['Scheduled_Travel_Time_min']} min")
                st.write(f"**Weather:** {weather}")
                st.write(f"**Previous Train Impact:** {row['Previous_Train_Delay_min']} min")


def show_chat_page(bot):
    """Customer care chatbot interface"""
    st.markdown("## 💬 Customer Care Assistant")
    st.markdown("Ask me anything about train delays, weather impacts, or travel recommendations!")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! 🙏 Welcome to Indian Railways Customer Care. I'm here to help you with:\n\n"
                          "• Train delay predictions\n"
                          "• Weather impact information\n"
                          "• Best travel time recommendations\n"
                          "• Real-time network status\n\n"
                          "How may I assist you today?"
            }
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        st.markdown(
            format_chat_message(message["role"], message["content"]),
            unsafe_allow_html=True
        )
    
    # Chat input
    user_input = st.chat_input("Type your question here... (e.g., 'Will morning fog affect my train?')")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(format_chat_message("user", user_input), unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤖 Thinking..."):
            # Parse user intent and extract context
            user_lower = user_input.lower()
            
            # Extract time context
            hour = datetime.now().hour
            if 'morning' in user_lower or 'am' in user_lower:
                hour = 8
            elif 'evening' in user_lower or 'pm' in user_lower:
                hour = 18
            elif 'afternoon' in user_lower:
                hour = 14
            
            # Extract weather context
            weather = 'Clear'
            is_fog = False
            if 'fog' in user_lower:
                weather = 'Fog'
                is_fog = True
            elif 'rain' in user_lower:
                weather = 'Rain'
            elif 'storm' in user_lower:
                weather = 'Storm'
            
            # Determine query type and respond
            if 'delay' in user_lower or 'late' in user_lower or 'time' in user_lower:
                conditions = {
                    'weather': weather,
                    'is_fog': is_fog,
                    'hour': hour,
                    'previous_delay': 10,
                    'track_maintenance': False,
                    'signal_failure': False
                }
                response = bot.respond('delay', conditions=conditions, train='your train')
            
            elif 'fog' in user_lower:
                response = bot.respond('fog', hour=hour)
            
            elif 'rain' in user_lower:
                response = bot.respond('weather_rain', hour=hour)
            
            elif 'morning' in user_lower or 'best time' in user_lower:
                response = bot.respond('morning')
            
            else:
                response = bot.respond('chat', message=user_input, hour=hour)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(format_chat_message("assistant", response), unsafe_allow_html=True)
        
        # Rerun to update chat
        st.rerun()
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("### 🎯 Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌫️ Morning Fog Impact"):
            st.session_state.messages.append({"role": "user", "content": "How does morning fog affect trains?"})
            st.rerun()
    
    with col2:
        if st.button("⏰ Best Travel Time"):
            st.session_state.messages.append({"role": "user", "content": "What's the best time for morning travel?"})
            st.rerun()
    
    with col3:
        if st.button("🌧️ Rain Delay Info"):
            st.session_state.messages.append({"role": "user", "content": "Will rain affect my train?"})
            st.rerun()


def show_analytics_page(bot):
    """Network analytics dashboard"""
    st.markdown("## 📊 Network Analytics")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_weather = st.multiselect(
            "Weather Conditions",
            options=bot.df['WeatherCondition'].unique(),
            default=bot.df['WeatherCondition'].unique()[:3]
        )
    
    with col2:
        delay_threshold = st.slider("Delay Threshold (min)", 0, 120, 30)
    
    with col3:
        selected_day = st.multiselect(
            "Day of Week",
            options=bot.df['Day_of_Week'].unique(),
            default=bot.df['Day_of_Week'].unique()
        )
    
    # Filter data
    filtered_df = bot.df[
        (bot.df['WeatherCondition'].isin(selected_weather)) &
        (bot.df['Day_of_Week'].isin(selected_day))
    ]
    
    st.markdown(f"### 📈 Showing {len(filtered_df):,} records")
    
    # Delay distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            filtered_df,
            x='Arrival_Delay_min',
            nbins=50,
            title="Delay Distribution",
            labels={'Arrival_Delay_min': 'Delay (minutes)', 'count': 'Frequency'},
            color_discrete_sequence=['#FF5733']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_df,
            x='WeatherCondition',
            y='Arrival_Delay_min',
            title="Delay by Weather Condition",
            labels={'Arrival_Delay_min': 'Delay (minutes)'},
            color='WeatherCondition'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cascade analysis
    st.markdown("### 🔄 Cascading Delay Analysis")
    
    fig = px.scatter(
        filtered_df.sample(min(1000, len(filtered_df))),
        x='Previous_Train_Delay_min',
        y='Arrival_Delay_min',
        color='WeatherCondition',
        size='Distance_km',
        title="Previous Train Delay vs Current Delay",
        labels={
            'Previous_Train_Delay_min': 'Previous Train Delay (min)',
            'Arrival_Delay_min': 'Current Train Delay (min)'
        },
        hover_data=['Distance_km', 'Day_of_Week']
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("### 📊 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Delay", f"{filtered_df['Arrival_Delay_min'].mean():.1f} min")
    
    with col2:
        st.metric("Median Delay", f"{filtered_df['Arrival_Delay_min'].median():.1f} min")
    
    with col3:
        st.metric("Max Delay", f"{filtered_df['Arrival_Delay_min'].max():.0f} min")
    
    with col4:
        ontime = (filtered_df['Arrival_Delay_min'] <= 5).sum() / len(filtered_df) * 100
        st.metric("On-Time %", f"{ontime:.1f}%")


def show_predictor_page(bot):
    """Delay predictor tool"""
    st.markdown("## ⚙️ Train Delay Predictor")
    st.markdown("Configure conditions to get AI-powered delay predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌤️ Weather Conditions")
        
        weather = st.selectbox(
            "Current Weather",
            options=['Clear', 'Rain', 'Fog', 'Storm', 'Clouds', 'Haze']
        )
        
        temp = st.slider("Temperature (°C)", 0, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 70)
        visibility = st.slider("Visibility (km)", 0, 20, 15)
    
    with col2:
        st.markdown("### 🚂 Train Conditions")
        
        hour = st.slider("Departure Hour", 0, 23, 8)
        prev_delay = st.number_input("Previous Train Delay (min)", 0, 180, 10)
        distance = st.number_input("Distance (km)", 0, 3000, 500)
        
        track_maint = st.checkbox("Track Maintenance Active")
        signal_fail = st.checkbox("Signal Failure Reported")
    
    # Predict button
    if st.button("🔮 Predict Delay", use_container_width=True):
        with st.spinner("Analyzing network conditions..."):
            # Build conditions
            conditions = {
                'weather': weather,
                'is_fog': (weather == 'Fog'),
                'hour': hour,
                'previous_delay': prev_delay,
                'track_maintenance': track_maint,
                'signal_failure': signal_fail,
                'distance': distance
            }
            
            # Get prediction
            scenarios, base_delay, similar = bot.predict_delay(conditions)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Delay card
            primary_delay = int(scenarios[0][1])
            primary_prob = int(scenarios[0][0] * 100)
            
            if primary_delay <= 10:
                color = "#4CAF50"
                status = "ON TIME ✅"
            elif primary_delay <= 30:
                color = "#FF9800"
                status = "MINOR DELAY ⚠️"
            else:
                color = "#F44336"
                status = "MAJOR DELAY 🚨"
            
            st.markdown(f"""
            <div style='background: {color}; padding: 30px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;'>
                <h1 style='margin: 0; font-size: 3em;'>{primary_delay} min</h1>
                <h2 style='margin: 10px 0;'>{status}</h2>
                <p style='margin: 0; font-size: 1.2em;'>Confidence: {primary_prob}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Scenarios chart
            st.plotly_chart(create_delay_visualization(scenarios), use_container_width=True)
            
            # RAG insights
            if similar:
                st.markdown("### 🔍 Historical Context (RAG Analysis)")
                st.info(f"Found {len(similar)} similar historical cases with average delay of {np.mean([s['delay'] for s in similar]):.0f} minutes")
                
                # Show similar cases
                with st.expander("View Similar Historical Cases"):
                    for i, case in enumerate(similar[:5], 1):
                        st.write(f"{i}. {case['node']}: {case['delay']:.1f} min delay")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if primary_delay <= 10:
                st.success("✅ Your train is expected to run smoothly. Have a pleasant journey!")
            elif primary_delay <= 30:
                st.warning("⚠️ Minor delay expected. Consider arriving at the platform a bit early.")
            else:
                st.error("🚨 Significant delay anticipated. Consider alternative arrangements if possible.")
                
                if weather == 'Fog' and hour < 11:
                    st.info("💡 Morning fog typically clears by 11 AM. Later trains may have better punctuality.")
                
                if primary_delay >= 60:
                    st.info("📱 SMS alerts and refreshments will be provided. Customer care: 139")


if __name__ == "__main__":
    main()
