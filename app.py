import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, compute_metrics

# --- Optimized Plotly Configuration ---
# Disable unnecessary features to reduce CPU/memory overhead
PLOTLY_CONFIG = {
    'responsive': True,
    'displayModeBar': False,  # Hide modebar to reduce overhead
    'staticPlot': False,  # Keep interactive but optimized
}

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
    /* Global Font - shadcn minimal style with Geist Sans */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Roboto", "Helvetica Neue", Arial, sans-serif !important;
        font-weight: 400;
        letter-spacing: -0.011em;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
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
        background-color: #020817;
        color: #f8fafc;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        color: #f8fafc;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: -0.006em;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.025em;
    }
        border-bottom: 1px solid #2d2f3b;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9ca3af;
        font-weight: 500;
        font-size: 0.95rem;
        padding-bottom: 12px;
        letter-spacing: -0.011em;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #ffffff;
        border-bottom: 2px solid #3b82f6;
        font-weight: 600;
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
    
    /* Headers - minimal style */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.25rem !important; }
    
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
# Load data once and pre-compute all metrics
with st.spinner('Loading Application Data...'):
    flights, airlines, airports = load_data()

if flights is None or flights.empty:
    st.error("Could not load data. Please ensure CSV files are present in the directory.")
    st.stop()

# Pre-compute all metrics once - this is cached and reused across all tabs
with st.spinner('Loading Data...'):
    metrics = compute_metrics(flights)

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
        autosize=True, # Re-enabled for autoscale
        # Removed animation transition to reduce CPU usage
        # transition={'duration': 500, 'easing': 'cubic-in-out'}
    )
    return fig

def create_gauge_chart(value, title, max_val=None, color="#f97316", unit="min"):
    if max_val is None:
        max_val = value * 2 if value > 0 else 100
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'font': {'size': 36, 'color': 'white'}, 'suffix': f' {unit}'},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75},
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
    
    # Add centered title as annotation
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=14, color='white', family='Inter')
    )
    
    fig.update_layout(
        height=240,
        margin={'t': 70, 'b': 20, 'l': 30, 'r': 30},
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': 'Inter'}
    )
    return fig

# Tabs Reorganization:# --- Layout Definitions ---
tab_delay, tab_time, tab_airline, tab_airport, tab_eda = st.tabs(["Delay Analysis", "Time Analysis", "Airline Analysis", "Airport Analysis", "Deep Dive (EDA)"])


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
def render_kpi_header(metrics_dict):
    """Render KPI header using pre-computed metrics"""
    total_airlines = metrics_dict['total_airlines']
    total_airports = metrics_dict['total_airports']
    total_flights = metrics_dict['total_flights']
    
    ot_count = metrics_dict['on_time_flights']
    dly_count = metrics_dict['delayed_flights']
    cnl_count = metrics_dict['cancelled_flights']
    
    ot_pct = metrics_dict['on_time_pct']
    dly_pct = metrics_dict['delayed_pct']
    cnl_pct = metrics_dict['cancelled_pct']
    
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

# --- Tab: Delay Analysis ---
with tab_delay:
    st.markdown("### Delay Analysis")
    
    render_kpi_header(metrics)
        
    # Use pre-computed delay metrics
    avg_airline_delay = metrics['avg_airline_delay']
    avg_aircraft_delay = metrics['avg_aircraft_delay']
    avg_system_delay = metrics['avg_system_delay']
    avg_weather_delay = metrics['avg_weather_delay']
    avg_security_delay = metrics['avg_security_delay']
    
    # 5 Gauge Charts in a row
    dg1, dg2, dg3, dg4, dg5 = st.columns(5)
    
    dg1.plotly_chart(create_gauge_chart(avg_airline_delay, "Average Airline Delay", max_val=30, color="#f97316"), width="stretch")
    dg2.plotly_chart(create_gauge_chart(avg_aircraft_delay, "Average Aircraft Delay", max_val=30, color="#f97316"), width="stretch")
    dg3.plotly_chart(create_gauge_chart(avg_system_delay, "Average System Delay", max_val=30, color="#f97316"), width="stretch")
    dg4.plotly_chart(create_gauge_chart(avg_weather_delay, "Average Weather Delay", max_val=10, color="#f97316"), width="stretch")
    dg5.plotly_chart(create_gauge_chart(avg_security_delay, "Average Security Delay", max_val=5, color="#f97316"), width="stretch")
    
    st.markdown("---")
        
    # Chart 1 & 2
    dc_c1, dc_c2 = st.columns(2)
    
    # Use pre-computed delay means by month
    delay_means_month = metrics['delay_means_month']
    
    with dc_c1:
        # Chart 1: Avg Airline & Aircraft Delay and Avg Air System Delay by Month (Stacked Bar)
        fig_c1 = px.bar(delay_means_month, x='Month', y=['Avg Airline & Aircraft Delay', 'Avg Air System Delay'],
                        title="Avg Airline & Aircraft Delay and Avg Air System Delay by Month",
                        color_discrete_map={'Avg Airline & Aircraft Delay': '#3b82f6', 'Avg Air System Delay': '#1e3a8a'})
        fig_c1 = update_chart_layout(fig_c1)
        fig_c1.update_layout(autosize=True, legend_title="", height=None)
        st.plotly_chart(fig_c1, width="stretch", config=PLOTLY_CONFIG)
        
    with dc_c2:
        # Chart 2: Avg Weather, Dep, Arr, Taxi Out Delay by Month (Stacked Bar)
        fig_c2 = px.bar(delay_means_month, x='Month', 
                        y=['WEATHER_DELAY', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT'],
                        title="Avg Weather, Dep, Arr, Taxi Out Delay by Month",
                        labels={'WEATHER_DELAY': 'Avg Weather Delay', 'DEPARTURE_DELAY': 'Avg Departure Delay', 'ARRIVAL_DELAY': 'Avg Arrival Delay', 'TAXI_OUT': 'Avg Taxi Out Delay'},
                        color_discrete_map={'WEATHER_DELAY': '#ef4444', 'DEPARTURE_DELAY': '#db2777', 'ARRIVAL_DELAY': '#f97316', 'TAXI_OUT': '#eab308'}
                        )
        fig_c2 = update_chart_layout(fig_c2)
        fig_c2.update_layout(autosize=True, legend_title="", height=None)
        st.plotly_chart(fig_c2, width="stretch", config=PLOTLY_CONFIG)
        
    st.markdown("---")
    
    # Chart 3: Average Delay by Airline (Monthwise) - Multi-line Area
    avg_delay_airline_month = metrics['avg_delay_airline_month']
    
    # Define distinct color palette for airlines (14 unique colors)
    airline_colors = [
        '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
        '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
        '#14b8a6', '#f43f5e', '#22d3ee', '#a855f7'
    ]
    
    # Create color map for each unique airline
    unique_airlines = avg_delay_airline_month['AIRLINE_NAME'].unique()
    color_map = {airline: airline_colors[i % len(airline_colors)] for i, airline in enumerate(unique_airlines)}
    
    # Reverting to Area chart as requested
    fig_c3 = px.area(avg_delay_airline_month, x='Month', y='ARRIVAL_DELAY', color='AIRLINE_NAME',
                    title="Average Delay by Airline (Monthwise)",
                    labels={'ARRIVAL_DELAY': 'Average Delay'},
                    color_discrete_map=color_map)
    fig_c3.update_traces(line_shape='spline')
    fig_c3.update_traces(stackgroup=None, fill='tozeroy') 
    fig_c3 = update_chart_layout(fig_c3)
    fig_c3.update_layout(autosize=True, width=None, height=None, showlegend=True, legend=dict(orientation="h", y=-0.25, x=0.5, xanchor='center'), margin=dict(l=10, r=10, t=40, b=150))
    with st.container():
        st.plotly_chart(fig_c3, width="stretch", config=PLOTLY_CONFIG)

# --- Tab: Time Analysis ---
with tab_time:
    st.markdown("### Temporal Flight Analysis")
    
    render_kpi_header(metrics)
    
    # Use pre-computed time stats
    month_stats = metrics['month_stats']
    dow_stats = metrics['dow_stats']
    day_stats = metrics['day_stats']
    delay_stream = metrics['delay_stream']
    
    # Prepare melted data for charts
    melted_month = month_stats.melt(id_vars=['Month Name'], value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Status', value_name='Count')
    melted_dow = dow_stats.melt(id_vars=['Day Name'], value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Status', value_name='Count')
    
    # Delay type melted data
    delay_melt = delay_stream.melt(id_vars=['Month'], var_name='Type', value_name='Minutes')
    type_map = {
        'AIR_SYSTEM_DELAY': 'Air System', 'LATE_AIRCRAFT_DELAY': 'Aircraft Delay',
        'AIRLINE_DELAY': 'Airline Delay', 'SECURITY_DELAY': 'Security',
        'TAXI_OUT': 'Taxi Out', 'WEATHER_DELAY': 'Weather'
    }
    delay_melt['Type'] = delay_melt['Type'].map(type_map)

    # Common Colors for Status
    colors_status = {'On Time': '#22c55e', 'Delayed': '#facc15', 'Cancelled': '#ef4444'}
    
    # Layout Grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        # Chart 1: Flight Analysis by Month (Stream)
        fig_m = px.area(melted_month, x='Month Name', y='Count', color='Status', title="Flight Analysis by Month",
                        color_discrete_map=colors_status)
        fig_m.update_traces(line_shape='spline')
        fig_m = update_chart_layout(fig_m)
        fig_m.update_layout(autosize=True, height=None)
        st.plotly_chart(fig_m, width="stretch", config=PLOTLY_CONFIG)
        
    with r1c2:
        # Chart 2: Flight Analysis by DOW (High to Low)
        fig_dow = px.area(melted_dow, x='Day Name', y='Count', color='Status', title="Flight Analysis by Day of Week (High to Low)",
                          color_discrete_map=colors_status)
        fig_dow.update_traces(line_shape='spline') 
        fig_dow = update_chart_layout(fig_dow)
        fig_dow.update_layout(autosize=True, height=None)
        st.plotly_chart(fig_dow, width="stretch", config=PLOTLY_CONFIG)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        # Chart 3: Flight Analysis by Days (Line)
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['Total'], name='Total Flights', line=dict(color='#3b82f6', width=3)))
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['On Time'], name='On Time Flight', line=dict(color='#22c55e', width=2)))
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['Delayed'], name='Delayed Flight', line=dict(color='#facc15', width=2)))
        fig_d.add_trace(go.Scatter(x=day_stats['DAY'], y=day_stats['Cancelled'], name='Cancelled Flight', line=dict(color='#ef4444', width=2)))
        
        fig_d.update_layout(title="Flight Analysis by Days", xaxis_title="Day of Month", autosize=True, height=None,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
        fig_d.update_xaxes(showgrid=False)
        fig_d.update_yaxes(showgrid=False)
        st.plotly_chart(fig_d, width="stretch", config=PLOTLY_CONFIG)

    with r2c2:
        # Chart 4: Delay Type Analysis by Month (Stream)
        fig_stream = px.area(delay_melt, x='Month', y='Minutes', color='Type', title="Delay Type Analysis by Month")
        fig_stream.update_traces(line_shape='spline')
        fig_stream = update_chart_layout(fig_stream)
        fig_stream.update_layout(autosize=True, height=None)
        st.plotly_chart(fig_stream, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("---")


with tab_airline:
    st.markdown("### Airline Analysis")

    # Use pre-computed airline metrics
    aa_delay_count = metrics['aa_delay_count']
    aa_delay_pct = metrics['aa_delay_pct']
    avg_dep = metrics['avg_dep']
    avg_arr = metrics['avg_arr']
    avg_aa_delay = metrics['avg_aa_delay']
    
    render_kpi_header(metrics)
    
    # --- Metrics & Gauges Row ---
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
        g1.plotly_chart(create_gauge_chart(avg_dep, "Average Departure Delay", max_val=30, color="#f97316"), width="stretch")
        g2.plotly_chart(create_gauge_chart(avg_arr, "Average Arrival Delay", max_val=30, color="#f97316"), width="stretch")
        g3.plotly_chart(create_gauge_chart(avg_aa_delay, "Average Airline & A/C Delay", max_val=15, color="#f97316"), width="stretch")

    st.markdown("---")
    
    # Row 2: Flight Analysis by Airline (Stacked Bar)
    airline_counts = metrics['airline_counts']
    
    airline_melt = airline_counts.melt(id_vars=['AIRLINE_NAME'], value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Status', value_name='Count')
    
    fig_airline_stack = px.bar(airline_melt, x='AIRLINE_NAME', y='Count', color='Status', 
                               title="Flight Analysis by Airline", 
                               color_discrete_map={'On Time': '#22c55e', 'Delayed': '#facc15', 'Cancelled': '#ef4444'})
    fig_airline_stack = update_chart_layout(fig_airline_stack)
    fig_airline_stack.update_layout(autosize=True, width=None, height=None, showlegend=True, legend=dict(orientation="h", y=-0.25, x=0.5, xanchor='center'), margin=dict(l=10, r=10, t=40, b=150))
    st.plotly_chart(fig_airline_stack, width="stretch", config=PLOTLY_CONFIG)
    
    st.markdown("---")
    
    # Row 3: Ranked Rate Charts
    c1, c2, c3 = st.columns(3)
    
    def create_rank_chart(df, col, color, title):
        top_10 = df.nlargest(10, col)
        fig = px.bar(top_10, y='AIRLINE_NAME', x=col, orientation='h', title=title, text=col)
        fig.update_traces(marker_color=color, texttemplate='%{text}%', textposition='inside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False) 
        fig = update_chart_layout(fig)
        fig.update_layout(height=None, margin=dict(l=180, r=10, t=40, b=10))
        return fig

    with c1:
        st.plotly_chart(create_rank_chart(airline_counts, 'On Time %', '#22c55e', "On Time Flights % by Airline"), width="stretch", config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(create_rank_chart(airline_counts, 'Delayed %', '#f97316', "Delayed Flights % by Airline"), width="stretch", config=PLOTLY_CONFIG)
    with c3:
        st.plotly_chart(create_rank_chart(airline_counts, 'Cancelled %', '#ef4444', "Cancelled Flights % by Airline"), width="stretch", config=PLOTLY_CONFIG)


# --- Tab: Airport Analysis ---
with tab_airport:
    st.markdown("### Airport Analysis")
    
    render_kpi_header(metrics)
    
    # Use pre-computed airport counts
    airport_counts = metrics['airport_counts']
    
    # Layout: Flight Analysis (Top - Full Width), Decomposition Trees (Bottom - 3 Cols)
    
    # 1. Main Chart (Top)
    # Sort for Main Chart (by Total Volume)
    airport_counts_vol = airport_counts.sort_values('Total', ascending=False).head(20)
    
    melted_ap = airport_counts_vol.melt(id_vars='ORIGIN_AIRPORT', value_vars=['On Time', 'Delayed', 'Cancelled'], var_name='Type', value_name='Count')
    fig_main = px.bar(melted_ap, x='ORIGIN_AIRPORT', y='Count', color='Type', title="Flight Analysis by Airport (Top 20)",
                       color_discrete_map={'On Time': '#22c55e', 'Delayed': '#facc15', 'Cancelled': '#ef4444'})
    fig_main = update_chart_layout(fig_main)

    fig_main.update_layout(autosize=True, width=None, height=None, showlegend=True, legend=dict(orientation="h", y=-0.25, x=0.5, xanchor='center'), margin=dict(l=10, r=10, t=40, b=150))
    with st.container():
        st.plotly_chart(fig_main, width="stretch", config=PLOTLY_CONFIG)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Helper for Decomposition Tree
    def plot_ap_tree(df, col_val, col_label, color, title):
         df_sorted = df.sort_values(col_val, ascending=False).head(10)
         fig = px.bar(df_sorted, y=col_label, x=col_val, orientation='h', title=title, text=col_val)
         fig.update_traces(marker_color=color, texttemplate='%{text:.1f}%', textposition='outside', width=0.6, cliponaxis=False)
         fig.update_layout(
             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
             font={'color': 'white', 'family': 'Inter'},
             xaxis={'visible': False}, 
             yaxis={'visible': True, 'showgrid': False, 'tickfont': {'size': 14}},
             margin=dict(l=10, r=80, t=40, b=10),
             height=350,
             showlegend=False
         )
         fig.update_yaxes(autorange="reversed")
         return fig

    # 2. Sub Charts (Bottom Row)
    c_ot, c_dly, c_cnl = st.columns(3)
    
    with c_ot:
         fig_ot = plot_ap_tree(airport_counts, 'On Time %', 'ORIGIN_AIRPORT', '#22c55e', "On Time % by Airport")
         st.plotly_chart(fig_ot, width="stretch", config=PLOTLY_CONFIG)
         
    with c_dly:
         fig_dly = plot_ap_tree(airport_counts, 'Delayed %', 'ORIGIN_AIRPORT', '#facc15', "Delayed % by Airport")
         st.plotly_chart(fig_dly, width="stretch", config=PLOTLY_CONFIG)
         
    with c_cnl:
         fig_cnl = plot_ap_tree(airport_counts, 'Cancelled %', 'ORIGIN_AIRPORT', '#ef4444', "Cancelled % by Airport")
         st.plotly_chart(fig_cnl, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("---")
# --- Tab: EDA ---
with tab_eda:
    st.markdown("### Deep Dive Analysis")
    
    st.markdown("#### 1. Flight Volume Patterns")
    st.markdown("**How does the overall flight volume vary by month? By day of week?**")
    
    q1_col1, q1_col2 = st.columns(2)
    
    with q1_col1:
        # Use pre-computed volume by month
        vol_month = metrics['vol_month']
        
        fig_vol_m = px.area(vol_month, x='Month Name', y='Count', title="Flight Volume by Month", markers=True)
        fig_vol_m.update_traces(line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.2)')
        fig_vol_m = update_chart_layout(fig_vol_m)
        st.plotly_chart(fig_vol_m, width="stretch", config=PLOTLY_CONFIG)
        
    with q1_col2:
        # Use pre-computed volume by day of week
        vol_day = metrics['vol_day']
        
        fig_vol_d = px.bar(vol_day, x='Day Name', y='Count', title="Flight Volume by Day of Week")
        fig_vol_d.update_traces(marker_color='#1d4ed8')
        fig_vol_d = update_chart_layout(fig_vol_d)
        fig_vol_d.update_layout(showlegend=False)
        st.plotly_chart(fig_vol_d, width="stretch", config=PLOTLY_CONFIG)
        
    st.markdown("---")

    st.markdown("#### 2. Departure Delay Insights")
    st.markdown("**Percetage of flights with departure delay (>15 mins) and average delay duration.**")

    # Use pre-computed departure delay metrics
    pct_dep_delayed = metrics['pct_dep_delayed']
    avg_dep_delay_duration = metrics['avg_dep_delay_duration']
    
    q2_c1, q2_c2 = st.columns(2)
    with q2_c1:
         st.metric("Departure Delayed %", f"{pct_dep_delayed:.2f}%")
    with q2_c2:
         st.metric("Avg Delay Duration (min)", f"{avg_dep_delay_duration:.1f}", help="Average minutes for flights that were delayed >15m")
         
    st.markdown("---")
    
    st.markdown("#### 3. Delay Trends: Overall vs Boston (BOS)")
    st.markdown("**How does the % of delayed flights vary throughout the year? Comparison with BOS.**")
    
    # Use pre-computed comparison trend
    comp_trend = metrics['comp_trend']
    
    fig_comp = px.line(comp_trend, x='Month', y='Delayed_Pct', color='Type', title="Departure Delay % Trends (National vs BOS)", markers=True,
                       color_discrete_map={'National Average': '#9ca3af', 'Boston (BOS)': '#22c55e'})
    fig_comp = update_chart_layout(fig_comp)
    with st.container():
        st.plotly_chart(fig_comp, width="stretch", config=PLOTLY_CONFIG)
    
    st.markdown("---")

    st.markdown("#### 4. Cancellation Analysis")
    st.markdown("**How many flights were cancelled in 2015? Why?**")
    
    # Use pre-computed cancellation metrics
    total_cancelled = metrics['total_cancelled']
    pct_cancelled = metrics['pct_cancelled']
    pct_weather = metrics['pct_weather']
    pct_airline = metrics['pct_airline']
    
    q4_c1, q4_c2, q4_c3 = st.columns(3)
    with q4_c1:
        st.metric("Total Cancelled", f"{total_cancelled:,}")
    with q4_c2:
         st.metric("% Due to Weather", f"{pct_weather:.1f}%")
    with q4_c3:
         st.metric("% Due to Airline", f"{pct_airline:.1f}%")