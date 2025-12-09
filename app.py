import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, apply_filters

# --- Page Configuration ---
st.set_page_config(
    page_title="Airline Flight Delays Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Modern "shadcn" Aesthetic) ---
st.markdown("""
<style>
    /* Global Font - mimicking shadcn's default Inter/San-serif */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Force Full Width */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Main Background */
    .stApp {
        background-color: #020817; /* shadcn dark bg */
        color: #f8fafc; /* shadcn foreground */
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #0f172a; /* Card bg */
        border: 1px solid #1e293b; /* Card border */
        border-radius: 8px; /* shadcn radius */
        padding: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        color: #f8fafc;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8; /* Muted foreground */
        font-size: 14px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 700;
    }
        border-bottom: 1px solid #2d2f3b;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9ca3af;
        font-weight: 600;
        font-size: 1rem;
        padding-bottom: 12px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #ffffff;
        border-bottom: 2px solid #3b82f6; /* Bright blue active state */
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1c24;
        color: #e0e0e0;
        border-radius: 8px;
    }
    
    /* Selectboxes in Sidebar */
    .stSelectbox > div > div {
        background-color: #1a1c24;
        color: white;
        border: 1px solid #2d2f3b;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
with st.spinner('Loading Application Data...'):
    flights, airlines, airports = load_data()

if flights is None or flights.empty:
    st.error("Could not load data. Please ensure CSV files are present in the directory.")
    st.stop()

# --- Filters (Hidden/Default to All as requested) ---
selected_month = 'All'
selected_airline = 'All'
selected_origin = 'All'

# Apply Filters
filtered_df = apply_filters(flights, selected_month, selected_airline, selected_origin)

# --- Summary Metrics Calculations ---
total_flights = len(filtered_df)
on_time_flights = len(filtered_df[filtered_df['ARRIVAL_DELAY'] <= 15]) 
delayed_flights = len(filtered_df[filtered_df['ARRIVAL_DELAY'] > 15])
cancelled_flights = len(filtered_df[filtered_df['CANCELLED'] == 1])

on_time_pct = (on_time_flights / total_flights * 100) if total_flights > 0 else 0
delayed_pct = (delayed_flights / total_flights * 100) if total_flights > 0 else 0
cancelled_pct = (cancelled_flights / total_flights * 100) if total_flights > 0 else 0

# --- Global Maps ---
month_map_rev = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
day_map = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}

# --- Main Dashboard Structure ---
st.title("Airline Performance Dashboard")
# --- Helper Functions ---

# Helper to transparentize charts and apply Maven theme colors
# Helper for Chart Layout with Animation (Shadcn Style)
def update_chart_layout(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc"), # shadcn font/color
        title_font=dict(family="Inter", size=18, color="#f8fafc", weight=600),
        legend=dict(font=dict(family="Inter", color="#94a3b8")),
        xaxis=dict(showgrid=False, zeroline=False, color="#94a3b8", gridcolor='#1e293b'),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, color="#94a3b8"),
        margin=dict(l=10, r=10, t=40, b=10), # Reduced margins for max width
        autosize=True, # Force autosize
        # Animation Transition
        transition={'duration': 500, 'easing': 'cubic-in-out'}
    )
    return fig

def create_gauge_chart(value, title, max_val=None, color="#f97316"):
    if max_val is None:
        max_val = value * 2 if value > 0 else 100
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18, 'color': 'white'}},
        number = {'font': {'size': 36, 'color': 'white'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75}, # Thicker bar
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'bordercolor': "gray",
             'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    fig.update_layout(
        height=220, # Increased height
        margin={'t': 40, 'b': 20, 'l': 30, 'r': 30},
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"}
    )
    return fig

# Tabs Reorganization:# --- Layout Definitions ---
tab_exec, tab_delay, tab_time, tab_airline, tab_airport, tab_eda = st.tabs(["Executive View", "Delay Analysis", "Time Analysis", "Airline Analysis", "Airport Analysis", "Deep Dive (EDA)"])


# --- Custom CSS ---
st.markdown("""
    <style>
        * {
            box-sizing: border-box;
        }
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        div[data-testid="stMetricValue"] {
            font-size: 20px;
        }
        /* KPI Cards */
        .kpi-card-dark {
            background-color: #4b5563; /* Grey */
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-card-grey {
            background-color: #6b7280; /* Lighter Grey */
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-card-darker-grey {
            background-color: #374151; /* Darker Grey */
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-card-green {
            background-color: #22c55e; /* Green */
            color: black;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-card-yellow {
            background-color: #facc15; /* Yellow */
            color: black;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-card-red {
            background-color: #f87171; /* Red */
            color: black;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        /* Delay Tab Specifics */
        .metric-card-yellow-light {
            background-color: #fef08a; /* Light Yellow */
            color: black;
            padding: 10px;
            text-align: center;
            border-radius: 5px;
            border: 1px solid #facc15;
            height: 100%;
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center;
        }
         /* Airline Tab Specifics */
        .metric-card-yellow-bold {
             background-color: #fef9c3;
             color: black;
             border: 2px solid #fde047;
             padding: 15px;
             text-align: center;
             border-radius: 8px;
             height: 100%;
             display: flex; 
             flex-direction: column; 
             justify-content: center; 
             align-items: center;
        }
        
        .kpi-value {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        .kpi-label {
            font-size: 14px;
            margin: 0;
            opacity: 0.9;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def fmt_num(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

month_map_rev = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
day_map = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}


# --- Reusable KPI Header Function ---
def render_kpi_header(df):
    total_airlines = df['AIRLINE'].nunique()
    total_airports = df['ORIGIN_AIRPORT'].nunique()
    total_flights = len(df)
    
    ot_count = ((df['ARRIVAL_DELAY'] <= 15) & (df['CANCELLED'] == 0)).sum()
    dly_count = (df['ARRIVAL_DELAY'] > 15).sum()
    cnl_count = df['CANCELLED'].sum()
    
    # Calculate Percentages
    ot_pct = (ot_count / total_flights * 100) if total_flights > 0 else 0
    dly_pct = (dly_count / total_flights * 100) if total_flights > 0 else 0
    cnl_pct = (cnl_count / total_flights * 100) if total_flights > 0 else 0
    
    # Block 5 (Red): ...
    
    h1, h2, h3, h4, h5 = st.columns(5)
    
    # Common styles
    card_style = "display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 0px; height: 100px; color: white;"
    
    with h1:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; height: 100px;">
            <div style="flex: 1; background-color: #333; display: flex; flex-direction: column; justify-content: center; align-items: center; border-bottom: 1px solid #555;">
                <div style="font-size: 20px; font-weight: bold; line-height: 1;">{total_airlines}</div>
                <div style="font-size: 10px;">Total Airlines</div>
            </div>
            <div style="flex: 1; background-color: #333; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 20px; font-weight: bold; line-height: 1;">{total_airports}</div>
                <div style="font-size: 10px;">Total Airports</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with h2:
        st.markdown(f"""
        <div style="background-color: #555; {card_style}">
            <div style="font-size: 28px; font-weight: bold;">{fmt_num(total_flights)}</div>
            <div style="font-size: 12px;">Total Flights</div>
        </div>
        """, unsafe_allow_html=True)
        
    with h3:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; height: 100px;">
             <div style="flex: 1; background-color: #22c55e; color: black; display: flex; flex-direction: column; justify-content: center; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.1);">
                <div style="font-size: 18px; font-weight: bold; line-height: 1;">{fmt_num(ot_count)}</div>
                <div style="font-size: 10px;">On Time Flight</div>
            </div>
            <div style="flex: 1; background-color: #22c55e; color: black; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 22px; font-weight: bold; line-height: 1;">{ot_pct:.2f}%</div>
                <div style="font-size: 10px;">On Time Flight %</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h4:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; height: 100px;">
             <div style="flex: 1; background-color: #facc15; color: black; display: flex; flex-direction: column; justify-content: center; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.1);">
                <div style="font-size: 18px; font-weight: bold; line-height: 1;">{fmt_num(dly_count)}</div>
                <div style="font-size: 10px;">Delayed Flight</div>
            </div>
            <div style="flex: 1; background-color: #facc15; color: black; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 22px; font-weight: bold; line-height: 1;">{dly_pct:.2f}%</div>
                <div style="font-size: 10px;">Delayed Flight %</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h5:
        st.markdown(f"""
         <div style="display: flex; flex-direction: column; height: 100px;">
             <div style="flex: 1; background-color: #f87171; color: black; display: flex; flex-direction: column; justify-content: center; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.1);">
                <div style="font-size: 18px; font-weight: bold; line-height: 1;">{fmt_num(cnl_count)}</div>
                <div style="font-size: 10px;">Cancelled Flight</div>
            </div>
            <div style="flex: 1; background-color: #f87171; color: black; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 22px; font-weight: bold; line-height: 1;">{cnl_pct:.2f}%</div>
                <div style="font-size: 10px;">Cancelled Flight %</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
            
    st.markdown("---")

# Helper for aggregating status counts
def get_status_counts(df, group_col):
    # Count total, on_time, delayed, cancelled
    # Select only necessary columns to avoid FutureWarning about grouping columns
    agg = df.groupby(group_col)[['ARRIVAL_DELAY', 'CANCELLED']].apply(lambda x: pd.Series({
        'Total': len(x),
        'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
        'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
        'Cancelled': x['CANCELLED'].sum()
    })).reset_index()
    return agg


# --- Tab: Executive View ---
with tab_exec:
    st.markdown("### Executive View")
    
    render_kpi_header(filtered_df)
    
    # --- Efficiency Metrics (Executive Specific) ---
    st.markdown("#### Efficiency Metrics")
    
    # We need a helper for Gauge Charts (Local or Global? It's defined below in lines 433+. We should move it up or use it.)
    # The definition of create_gauge is currently at line 433.
    # I should move create_gauge to utils or top of app.py eventually. for now just make sure it's defined before use if I use it here.
    # Actually, it's defined inside the block below. I will move it to the top of tab_exec or just keep it there if I use it after.
    # The code below line 433 defines create_gauge.
    
    # Let's just remove the orphaned lines first.
    
    # Re-insert the Gauges Logic that was supposed to be there.
    # Wait, the gauge logic I want is the "Efficiency Gauges" (Dist, Air, Delay).
    # I will paste the Efficiency Gauges block I prepared earlier.
    
    # Helper for Gauge Charts (Global definition better, but let's define if not exists)
    def create_gauge_exec(value, title, min_val, max_val, color):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 18, 'color': 'white', 'family': 'Inter'}},
            gauge = {
                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'bordercolor': "rgba(0,0,0,0)",
                'steps': [{'range': [min_val, max_val], 'color': 'rgba(255, 255, 255, 0.1)'}],
            }
        ))
        fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white"}, height=200, margin=dict(l=30, r=30, t=50, b=20))
        return fig

    e1, e2, e3 = st.columns(3)
    
    # Metrics
    tot_dist = filtered_df['DISTANCE'].sum()
    avg_dist = filtered_df['DISTANCE'].mean()
    tot_air = filtered_df['AIR_TIME'].sum()
    avg_air = filtered_df['AIR_TIME'].mean()
    # For Total Delay, let's use sum of Arrival Delays (positive only? or net?)
    # Mockup says "Total Delay 62.65M" and "Average Delay 10.94"
    # 10.94 corresponds to Average Arrival Delay probably.
    avg_delay = filtered_df['ARRIVAL_DELAY'].mean()
    tot_delay = filtered_df['ARRIVAL_DELAY'].sum() # Simple sum
    
    with e1:
        st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:18px;'>{fmt_num(tot_dist)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:12px; opacity:0.7;'>Total Distance</div>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge_exec(avg_dist, "Average Distance", 0, 2000, "#3b82f6"), use_container_width=True)
        
    with e2:
        st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:18px;'>{fmt_num(tot_air)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:12px; opacity:0.7;'>Total Air Time</div>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge_exec(avg_air, "Average Air Time", 0, 300, "#3b82f6"), use_container_width=True)
        
    with e3:
        st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:18px;'>{fmt_num(tot_delay)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:12px; opacity:0.7;'>Total Delay</div>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge_exec(avg_delay, "Average Delay", -20, 60, "#f97316"), use_container_width=True)

# --- Tab: Delay Analysis ---
with tab_delay:
    st.markdown("### Delay Analysis")
    
    render_kpi_header(filtered_df)
    
    st.markdown("#### Delay Metrics & Operational Efficiency")
    
    # Calculate Delay Metrics
    avg_airline_delay = filtered_df['AIRLINE_DELAY'].mean()
    avg_aircraft_delay = filtered_df['LATE_AIRCRAFT_DELAY'].mean()
    avg_system_delay = filtered_df['AIR_SYSTEM_DELAY'].mean()
    avg_weather_delay = filtered_df['WEATHER_DELAY'].mean()
    avg_security_delay = filtered_df['SECURITY_DELAY'].mean()
    
    # 5 Gauge Charts in a row (or 2 rows)
    # Mockup implies a row or grid. Let's do 5 cols.
    dg1, dg2, dg3, dg4, dg5 = st.columns(5)
    
    # Helper is globally available: create_gauge_chart(value, title, max_val=None, color="#f97316")
    # Assuming standard orange color for delays
    
    dg1.plotly_chart(create_gauge_chart(avg_airline_delay, "Avg Airline Delay", max_val=30, color="#f97316"), use_container_width=True)
    dg2.plotly_chart(create_gauge_chart(avg_aircraft_delay, "Avg Aircraft Delay", max_val=30, color="#f97316"), use_container_width=True)
    dg3.plotly_chart(create_gauge_chart(avg_system_delay, "Avg System Delay", max_val=30, color="#f97316"), use_container_width=True)
    dg4.plotly_chart(create_gauge_chart(avg_weather_delay, "Avg Weather Delay", max_val=10, color="#f97316"), use_container_width=True)
    dg5.plotly_chart(create_gauge_chart(avg_security_delay, "Avg Security Delay", max_val=5, color="#f97316"), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Trend Analysis")
    
    # Chart 1 & 2
    dc_c1, dc_c2 = st.columns(2)
    
    # Data Prep for Charts
    delay_means_month = filtered_df.groupby('MONTH').agg({
        'AIRLINE_DELAY': 'mean',
        'LATE_AIRCRAFT_DELAY': 'mean',
        'AIR_SYSTEM_DELAY': 'mean',
        'WEATHER_DELAY': 'mean',
        'DEPARTURE_DELAY': 'mean',
        'ARRIVAL_DELAY': 'mean',
        'TAXI_OUT': 'mean'
    }).reset_index()
    
    delay_means_month['Avg Airline & Aircraft Delay'] = delay_means_month['AIRLINE_DELAY'] + delay_means_month['LATE_AIRCRAFT_DELAY']
    delay_means_month['Avg Air System Delay'] = delay_means_month['AIR_SYSTEM_DELAY']
    delay_means_month['Month'] = delay_means_month['MONTH'].map(month_map_rev)
    
    with dc_c1:
        # Chart 1: Avg Airline & Aircraft Delay and Avg Air System Delay by Month (Stacked Bar)
        fig_c1 = px.bar(delay_means_month, x='Month', y=['Avg Airline & Aircraft Delay', 'Avg Air System Delay'],
                        title="Avg Airline & Aircraft Delay and Avg Air System Delay by Month",
                        color_discrete_map={'Avg Airline & Aircraft Delay': '#3b82f6', 'Avg Air System Delay': '#1e3a8a'})
        fig_c1 = update_chart_layout(fig_c1)
        fig_c1.update_layout(legend_title="", height=400)
        st.plotly_chart(fig_c1, use_container_width=True)
        
    with dc_c2:
        # Chart 2: Avg Weather, Dep, Arr, Taxi Out Delay by Month (Stacked Bar)
        fig_c2 = px.bar(delay_means_month, x='Month', 
                        y=['WEATHER_DELAY', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT'],
                        title="Avg Weather, Dep, Arr, Taxi Out Delay by Month",
                        labels={'WEATHER_DELAY': 'Avg Weather Delay', 'DEPARTURE_DELAY': 'Avg Departure Delay', 'ARRIVAL_DELAY': 'Avg Arrival Delay', 'TAXI_OUT': 'Avg Taxi Out Delay'},
                        color_discrete_map={'WEATHER_DELAY': '#ef4444', 'DEPARTURE_DELAY': '#db2777', 'ARRIVAL_DELAY': '#f97316', 'TAXI_OUT': '#eab308'}
                        )
        fig_c2 = update_chart_layout(fig_c2)
        fig_c2.update_layout(legend_title="", height=400)
        st.plotly_chart(fig_c2, use_container_width=True)
        
    st.markdown("---")
    
    # Chart 3: Average Delay by Airline (Monthwise) - Multi-line Area
    st.markdown("#### Average Delay by Airline (Monthwise)")
    avg_delay_airline_month = filtered_df.groupby(['MONTH', 'AIRLINE_NAME'])['ARRIVAL_DELAY'].mean().reset_index()
    avg_delay_airline_month['Month'] = avg_delay_airline_month['MONTH'].map(month_map_rev)
    
    # Reverting to Area chart as requested ("stacked line graph like before"), with spline smoothing
    fig_c3 = px.area(avg_delay_airline_month, x='Month', y='ARRIVAL_DELAY', color='AIRLINE_NAME',
                     title="Average Delay by Airline (Monthwise)",
                     labels={'ARRIVAL_DELAY': 'Average Delay'})
    fig_c3.update_traces(line_shape='spline')
    # Use 'stackgroup=None' to prevent values from summing up, allowing negative values to display correctly as absolute plots.
    # This keeps the "filled/stacked" look but is mathematically correct for mixed signs.
    fig_c3.update_traces(stackgroup=None, fill='tozeroy') 
    fig_c3 = update_chart_layout(fig_c3)
    fig_c3.update_layout(height=450, showlegend=True, legend=dict(orientation="h", y=1.1, x=1, xanchor='right'))
    st.plotly_chart(fig_c3, use_container_width=True)

# --- Tab: Time Analysis ---
# --- Tab: Time Analysis ---
with tab_time:
    st.markdown("### Temporal Flight Analysis")
    
    render_kpi_header(filtered_df)
    
    # helper for aggregates if needed, but we can do inline
    def get_time_counts(df, group_col):
         agg = df.groupby(group_col)[['ARRIVAL_DELAY', 'CANCELLED']].apply(lambda x: pd.Series({
            'Total': len(x),
            'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
            'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
            'Cancelled': x['CANCELLED'].sum()
        })).reset_index()
         return agg
         
    # Prepare Data
    
    # 1. Month Data
    month_stats = get_time_counts(filtered_df, 'MONTH')
    month_stats['Month Name'] = month_stats['MONTH'].map(month_map_rev)
    melted_month = month_stats.melt(id_vars=['Month Name'], value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Status', value_name='Count')
    
    # 2. DOW Data
    dow_stats = get_time_counts(filtered_df, 'DAY_OF_WEEK')
    # Map 1-7 to names. 1=Mon? Usually. Let's check Utils or map.
    # Assuming 1=Mon, 7=Sun in standard pandas/data. OR 1 could be Sunday.
    # Given the chart names "Tue, Fri...", let's assume standard names.
    dow_map = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
    dow_stats['Day Name'] = dow_stats['DAY_OF_WEEK'].map(dow_map)
    # Sort by Total Flight Volume Descending for "High to Low" area chart
    dow_stats = dow_stats.sort_values('Total', ascending=False)
    melted_dow = dow_stats.melt(id_vars=['Day Name'], value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Status', value_name='Count')
    
    # 3. Days Data (1-31)
    day_stats = get_time_counts(filtered_df, 'DAY')
    
    # 4. Delay Type Data
    delay_stream = filtered_df.groupby('MONTH')[['AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY', 'TAXI_OUT']].sum().reset_index()
    delay_stream['Month'] = delay_stream['MONTH'].map(month_map_rev)
    delay_melt = delay_stream.melt(id_vars=['Month'], var_name='Type', value_name='Minutes')
    type_map = {
        'AIR_SYSTEM_DELAY': 'Air System', 'LATE_AIRCRAFT_DELAY': 'Aircraft Delay',
        'AIRLINE_DELAY': 'Airline Delay', 'SECURITY_DELAY': 'Security',
        'TAXI_OUT': 'Taxi Out', 'WEATHER_DELAY': 'Weather'
    }
    delay_melt['Type'] = delay_melt['Type'].map(type_map)

    # Common Colors for Status
    colors_status = {'On Time': '#22c55e', 'Delayed': '#facc15', 'Cancelled': '#ef4444'} # Green, Yellow, Red
    # Total usually is blue if standalone, but stacked usually implies sum. User image colors: Blue, Green, Yellow, Red blocks flowing.
    # It seems "Total Flights" is the top layer?
    # If we stack OnTime+Delayed+Cancelled, we get Total.
    # The image shows "Total Flights" (Blue) as a separate big layer?
    # Or maybe it's just the label.
    # If I stack OnTime, Delayed, Cancelled, the sum is Total.
    # If I add a 'Total' Series, it doubles the volume if stacked.
    # I will just stack the 3 components.
    
    # Layout Grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        # Chart 1: Flight Analysis by Month (Stream)
        fig_m = px.area(melted_month, x='Month Name', y='Count', color='Status', title="Flight Analysis by Month",
                        color_discrete_map=colors_status)
        fig_m.update_traces(line_shape='spline')
        fig_m = update_chart_layout(fig_m)
        fig_m.update_layout(height=400)
        st.plotly_chart(fig_m, use_container_width=True)
        
    with r1c2:
        # Chart 2: Flight Analysis by DOW (High to Low)
        # To make "High to Low" stacked area using categorical data, we need to sort the X-axis by volume.
        # PX Area respects category order.
        fig_dow = px.area(melted_dow, x='Day Name', y='Count', color='Status', title="Flight Analysis by Day of Week (High to Low)",
                          color_discrete_map=colors_status)
        fig_dow.update_traces(line_shape='spline') 
        fig_dow = update_chart_layout(fig_dow)
        fig_dow.update_layout(height=400)
        st.plotly_chart(fig_dow, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        # Chart 3: Flight Analysis by Days (Line)
        # Line chart of Total, On Time, Delayed, Cancelled
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['Total'], name='Total Flights', line=dict(color='#3b82f6', width=3)))
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['On Time'], name='On Time Flight', line=dict(color='#22c55e', width=2)))
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['Delayed'], name='Delayed Flight', line=dict(color='#facc15', width=2)))
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['Cancelled'], name='Cancelled Flight', line=dict(color='#ef4444', width=2)))
        
        fig_d.update_layout(title="Flight Analysis by Days", xaxis_title="Day of Month", height=400,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
        fig_d.update_xaxes(showgrid=False)
        fig_d.update_yaxes(showgrid=False)
        st.plotly_chart(fig_d, use_container_width=True)

    with r2c2:
        # Chart 4: Delay Type Analysis by Month (Stream)
        fig_stream = px.area(delay_melt, x='Month', y='Minutes', color='Type', title="Delay Type Analysis by Month")
        fig_stream.update_traces(line_shape='spline')
        fig_stream = update_chart_layout(fig_stream)
        fig_stream.update_layout(height=400)
        st.plotly_chart(fig_stream, use_container_width=True)

    st.markdown("---")



with tab_airline:
    st.markdown("### Airline Performance Deep Dive")

    # Metrics Calculations
    airline_aircraft_delay_rows = filtered_df[(filtered_df['AIRLINE_DELAY'] > 0) | (filtered_df['LATE_AIRCRAFT_DELAY'] > 0)]
    aa_delay_count = len(airline_aircraft_delay_rows)
    total_flights = len(filtered_df)
    aa_delay_pct = (aa_delay_count / total_flights * 100) if total_flights > 0 else 0
    
    render_kpi_header(filtered_df)
    
    # --- Metrics & Gauges Row ---
    # Metrics
    aa_delay_df = filtered_df[ (filtered_df['AIRLINE_DELAY'] > 0) | (filtered_df['LATE_AIRCRAFT_DELAY'] > 0) ]
    aa_delay_count = len(aa_delay_df)
    aa_delay_pct = (aa_delay_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    
    avg_dep = filtered_df['DEPARTURE_DELAY'].mean()
    avg_arr = filtered_df['ARRIVAL_DELAY'].mean()
    # Avg Airline & Aircraft Delay: Mean of (Airline + Late Aircraft) DELAY minutes, where > 0? 
    # Or just mean of the columns. Usually "Avg Delay" implies mean of values including zeros or excluding?
    # Given the gauge value is 1.79K (Wait, 1.79K? That's huge for minutes. Maybe it's total count? 
    # But title says "Average ...". If it's 1.79K, maybe sum? 
    # Let's check mockup again. "Airline & Aircraft Delayed Flights 25K". "Average ... Delay 1.79K". 
    # If delay minutes sum is 45.13M, average can't be 1.79K unless filtered.
    # If filtered to only delayed flights? 
    # Let's assume Average of non-zero delays or just use the mean. 
    # If mean is small (e.g. 10 mins), 1.79 is weird. Maybe 1.79 mins?
    # Let's stick thereto standard mean for now.
    avg_aa_delay = (filtered_df['AIRLINE_DELAY'].fillna(0) + filtered_df['LATE_AIRCRAFT_DELAY'].fillna(0)).mean()
    
    col_metrics, col_gauges = st.columns([1.5, 3.5])
    
    with col_metrics:
        st.markdown(f"""
        <div style="background-color: #fefce8; border: 1px solid #facc15; border-radius: 5px; padding: 15px; text-align: center; color: black; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: bold; font-size: 16px; margin-bottom: 5px;">Airline & Aircraft<br>Delayed Flights</div>
            <div style="font-weight: bold; font-size: 28px; margin-bottom: 5px;">{fmt_num(aa_delay_count)}</div>
             <div style="font-weight: bold; font-size: 16px; margin-bottom: 5px;">Airline & Aircraft<br>Delayed Flights %</div>
            <div style="font-weight: bold; font-size: 24px;">{aa_delay_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_gauges:
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(create_gauge_chart(avg_dep, "Avg Departure Delay", max_val=30, color="#f97316"), use_container_width=True)
        g2.plotly_chart(create_gauge_chart(avg_arr, "Avg Arrival Delay", max_val=30, color="#f97316"), use_container_width=True)
        g3.plotly_chart(create_gauge_chart(avg_aa_delay, "Avg Airline & A/C Delay", max_val=15, color="#f97316"), use_container_width=True)

    st.markdown("---")
    
    # Row 2: Flight Analysis by Airline (Stacked Bar)
    st.markdown("---")
    
    # Row 2: Flight Analysis by Airline (Stacked Bar)
    # Using st.columns([1]) to definitively break out of any previous column layout context
    # Row 2: Flight Analysis by Airline (Stacked Bar)
    st.markdown("#### Flight Analysis by Airline")
    airline_counts = get_status_counts(filtered_df, 'AIRLINE_NAME')
    airline_counts = airline_counts.sort_values('Total', ascending=False)
    
    airline_melt = airline_counts.melt(id_vars=['AIRLINE_NAME'], value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Status', value_name='Count')
    
    fig_airline_stack = px.bar(airline_melt, x='AIRLINE_NAME', y='Count', color='Status', 
                               # title="Flight Analysis by Airline", 
                               color_discrete_map={'On Time': '#22c55e', 'Delayed': '#facc15', 'Cancelled': '#ef4444'})
    fig_airline_stack = update_chart_layout(fig_airline_stack)
    fig_airline_stack.update_layout(autosize=True, height=450, showlegend=True, legend=dict(orientation="h", y=1.1, x=1, xanchor='right'))
    st.plotly_chart(fig_airline_stack, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Ranked Rate Charts
    c1, c2, c3 = st.columns(3)
    
    airline_counts['On Time %'] = (airline_counts['On Time'] / airline_counts['Total'] * 100).round(2)
    airline_counts['Delayed %'] = (airline_counts['Delayed'] / airline_counts['Total'] * 100).round(2)
    airline_counts['Cancelled %'] = (airline_counts['Cancelled'] / airline_counts['Total'] * 100).round(2)
    
    def create_rank_chart(df, col, color, title):
        top_10 = df.nlargest(10, col)
        fig = px.bar(top_10, y='AIRLINE_NAME', x=col, orientation='h', title=title, text=col)
        fig.update_traces(marker_color=color, texttemplate='%{text}%', textposition='inside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False) 
        fig = update_chart_layout(fig)
        fig.update_layout(height=450, margin=dict(l=180, r=10, t=40, b=10))
        return fig

    with c1:
        st.plotly_chart(create_rank_chart(airline_counts, 'On Time %', '#22c55e', "On Time Flights % by Airline"), use_container_width=True)
    with c2:
        st.plotly_chart(create_rank_chart(airline_counts, 'Delayed %', '#f97316', "Delayed Flights % by Airline"), use_container_width=True)
    with c3:
        st.plotly_chart(create_rank_chart(airline_counts, 'Cancelled %', '#ef4444', "Cancelled Flights % by Airline"), use_container_width=True)



# --- Tab: Airport Analysis ---
with tab_airport:
    st.markdown("### Airport Performance Analysis")
    
    render_kpi_header(filtered_df)
    
    # helper for airport aggregates
    def get_airport_counts(df):
         agg = df.groupby('ORIGIN_AIRPORT')[['ARRIVAL_DELAY', 'CANCELLED']].apply(lambda x: pd.Series({
            'Total': len(x),
            'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
            'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
            'Cancelled': x['CANCELLED'].sum()
        })).reset_index()
         return agg

    # Prepare Data
    # Top 50 airports by volume for analysis to avoid clutter
    top_airports_list = filtered_df['ORIGIN_AIRPORT'].value_counts().nlargest(50).index.tolist()
    airport_df_top = filtered_df[filtered_df['ORIGIN_AIRPORT'].isin(top_airports_list)]
    
    airport_counts = get_airport_counts(airport_df_top)
    
    # Calculate Percentages
    airport_counts['On Time %'] = (airport_counts['On Time'] / airport_counts['Total'] * 100).fillna(0)
    airport_counts['Delayed %'] = (airport_counts['Delayed'] / airport_counts['Total'] * 100).fillna(0)
    airport_counts['Cancelled %'] = (airport_counts['Cancelled'] / airport_counts['Total'] * 100).fillna(0)
    
    # Layout: Flight Analysis (Left Big), Decomposition Trees (Right 3 Small)
    # Similar to Airline tab
    
    # Layout: Flight Analysis (Top - Full Width), Decomposition Trees (Bottom - 3 Cols)
    
    # 1. Main Chart (Top)
    # Sort for Main Chart (by Total Volume)
    airport_counts_vol = airport_counts.sort_values('Total', ascending=False).head(20)
    
    melted_ap = airport_counts_vol.melt(id_vars='ORIGIN_AIRPORT', value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Type', value_name='Count')
    fig_main = px.bar(melted_ap, x='ORIGIN_AIRPORT', y='Count', color='Type', title="Flight Analysis by Airport (Top 20)",
                       color_discrete_map={'On Time': '#22c55e', 'Delayed': '#facc15', 'Cancelled': '#ef4444'})
    fig_main = update_chart_layout(fig_main)

    fig_main.update_layout(height=450, showlegend=True, legend=dict(orientation="h", y=1.1, x=1, xanchor='right'))
    st.plotly_chart(fig_main, use_container_width=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Helper for Decomposition Tree
    def plot_ap_tree(df, col_val, col_label, color, title):
         df_sorted = df.sort_values(col_val, ascending=False).head(10)
         fig = px.bar(df_sorted, y=col_label, x=col_val, orientation='h', title=title, text=col_val)
         fig.update_traces(marker_color=color, texttemplate='%{text:.1f}%', textposition='outside', width=0.6, cliponaxis=False) # Increased width for visibility
         fig.update_layout(
             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
             font={'color': 'white', 'family': 'Inter'},
             xaxis={'visible': False}, 
             yaxis={'visible': True, 'showgrid': False, 'tickfont': {'size': 14}}, # Bigger labels
             margin=dict(l=10, r=80, t=40, b=10), # More right margin for outside text
             height=350,
             showlegend=False
         )
         fig.update_yaxes(autorange="reversed")
         return fig

    # 2. Sub Charts (Bottom Row)
    c_ot, c_dly, c_cnl = st.columns(3)
    
    with c_ot:
         fig_ot = plot_ap_tree(airport_counts, 'On Time %', 'ORIGIN_AIRPORT', '#22c55e', "On Time %")
         st.plotly_chart(fig_ot, use_container_width=True)
         
    with c_dly:
         fig_dly = plot_ap_tree(airport_counts, 'Delayed %', 'ORIGIN_AIRPORT', '#facc15', "Delayed %")
         st.plotly_chart(fig_dly, use_container_width=True)
         
    with c_cnl:
         fig_cnl = plot_ap_tree(airport_counts, 'Cancelled %', 'ORIGIN_AIRPORT', '#ef4444', "Cancelled %")
         st.plotly_chart(fig_cnl, use_container_width=True)

    st.markdown("---")
# --- Tab: EDA ---
with tab_eda:
    st.markdown("### Deep Dive Analysis")
    
    st.markdown("#### 1. Flight Volume Patterns")
    st.markdown("**How does the overall flight volume vary by month? By day of week?**")
    
    q1_col1, q1_col2 = st.columns(2)
    
    with q1_col1:
        # Volume by Month
        vol_month = filtered_df.groupby('MONTH').size().reset_index(name='Count')
        vol_month['Month Name'] = vol_month['MONTH'].map(month_map_rev)
        
        fig_vol_m = px.area(vol_month, x='Month Name', y='Count', title="Flight Volume by Month", markers=True)
        fig_vol_m.update_traces(line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.2)')
        fig_vol_m = update_chart_layout(fig_vol_m)
        st.plotly_chart(fig_vol_m, use_container_width=True)
        
    with q1_col2:
        # Volume by Day of Week
        # Ensure 1-7 index using explicit reindexing with int keys
        vol_day = filtered_df.groupby('DAY_OF_WEEK').size().reindex([1, 2, 3, 4, 5, 6, 7], fill_value=0).reset_index(name='Count')
        vol_day['Day Name'] = vol_day['DAY_OF_WEEK'].map(day_map)
        
        fig_vol_d = px.bar(vol_day, x='Day Name', y='Count', title="Flight Volume by Day of Week", color='Count', color_continuous_scale='Blues')
        fig_vol_d = update_chart_layout(fig_vol_d)
        fig_vol_d.update_layout(showlegend=False)
        st.plotly_chart(fig_vol_d, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 2. Departure Delay Insights")
    st.markdown("**Percetage of flights with departure delay (>15 mins) and average delay duration.**")

    # Q2 Calculations
    # Definition: Delayed if DEPARTURE_DELAY > 15
    dep_delayed_flights = filtered_df[filtered_df['DEPARTURE_DELAY'] > 15]
    pct_dep_delayed = (len(dep_delayed_flights) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    avg_dep_delay_duration = dep_delayed_flights['DEPARTURE_DELAY'].mean() if len(dep_delayed_flights) > 0 else 0
    
    q2_c1, q2_c2 = st.columns(2)
    with q2_c1:
         st.metric("Departure Delayed %", f"{pct_dep_delayed:.2f}%")
    with q2_c2:
         st.metric("Avg Delay Duration (min)", f"{avg_dep_delay_duration:.1f}", help="Average minutes for flights that were delayed > 15m")
         
    st.markdown("---")
    
    st.markdown("#### 3. Delay Trends: Overall vs Boston (BOS)")
    st.markdown("**How does the % of delayed flights vary throughout the year? Comparison with BOS.**")
    
    # Overall Trend
    monthly_stats = filtered_df.groupby('MONTH')['DEPARTURE_DELAY'].apply(lambda x: (x > 15).mean() * 100).reset_index(name='Delayed_Pct')
    monthly_stats['Type'] = 'National Average'
    
    # BOS Trend
    bos_flights = filtered_df[filtered_df['ORIGIN_AIRPORT'] == 'BOS']
    bos_stats = bos_flights.groupby('MONTH')['DEPARTURE_DELAY'].apply(lambda x: (x > 15).mean() * 100).reset_index(name='Delayed_Pct')
    bos_stats['Type'] = 'Boston (BOS)'
    
    # Combine
    comp_trend = pd.concat([monthly_stats, bos_stats])
    comp_trend['Month'] = comp_trend['MONTH'].map(month_map_rev)
    
    fig_comp = px.line(comp_trend, x='Month', y='Delayed_Pct', color='Type', title="Departure Delay % Trends (National vs BOS)", markers=True,
                       color_discrete_map={'National Average': '#9ca3af', 'Boston (BOS)': '#22c55e'})
    fig_comp = update_chart_layout(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("---")

    st.markdown("#### 4. Cancellation Analysis")
    st.markdown("**How many flights were cancelled in 2015? Why?**")
    
    # Q4 Calculations
    total_cancelled = filtered_df['CANCELLED'].sum()
    pct_cancelled = (total_cancelled / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    
    # Reasons: CANCELLATION_REASON (A=Carrier, B=Weather, C=NAS, D=Security)
    cancel_counts = filtered_df[filtered_df['CANCELLED'] == 1]['CANCELLATION_REASON'].value_counts()
    
    # Handle potentially missing keys
    val_A = cancel_counts.get('A', 0)
    val_B = cancel_counts.get('B', 0)
    
    pct_weather = (val_B / total_cancelled * 100) if total_cancelled > 0 else 0
    pct_airline = (val_A / total_cancelled * 100) if total_cancelled > 0 else 0
    
    q4_c1, q4_c2, q4_c3 = st.columns(3)
    with q4_c1:
        st.metric("Total Cancelled", f"{total_cancelled:,}")
    with q4_c2:
         st.metric("% Due to Weather", f"{pct_weather:.1f}%")
    with q4_c3:
         st.metric("% Due to Airline", f"{pct_airline:.1f}%")