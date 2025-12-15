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
        number = {
            'font': {'size': 36, 'color': 'white'}, 
            'suffix': f' {unit}',
            'valueformat': '.1f'
        },
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
    
    # Add centered title as annotation at the top
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.0,
        xanchor='center', yanchor='top',
        showarrow=False,
        font=dict(size=14, color='white', family='Inter')
    )
    
    fig.update_layout(
        height=240,
        margin={'t': 50, 'b': 20, 'l': 30, 'r': 30},
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': 'Inter'}
    )
    return fig

# Tabs Reorganization:# --- Layout Definitions ---
tab_delay, tab_time, tab_airline, tab_airport, tab_ml, tab_about = st.tabs([
    "Delay Analysis", "Time Analysis", "Airline Analysis", "Airport Analysis", 
    "ML Prediction", "About"
])


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
        fig_dow = px.area(melted_dow, x='Day Name', y='Count', color='Status', title="Flight Analysis by Day of Week",
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
# --- Tab: ML Prediction ---
with tab_ml:
    st.markdown("### Machine Learning: Flight Delay Prediction")
    
    # Try to load pre-trained model first
    from utils import load_trained_model, prepare_ml_data, train_delay_model, get_model_metrics, predict_delay_probability
    
    model_data = load_trained_model()
    
    if model_data is not None:
        # Use pre-trained model
        model = model_data['model']
        feature_names = model_data['feature_names']
        label_encoders = model_data['label_encoders']
        ml_metrics = model_data['metrics']
    else:
        # Train model on-the-fly (cached to avoid retraining)
        st.warning("⚠️ No pre-trained model found. Training model now... (This may take a minute)")
        st.info("💡 **Tip**: Run `python train_model.py` to pre-train the model and save it to disk for faster loading.")
        
        with st.spinner('Training ML model... (This only happens once per session)'):
            X_train, X_test, y_train, y_test, feature_names, label_encoders = prepare_ml_data(flights)
            model = train_delay_model(X_train, y_train)
            ml_metrics = get_model_metrics(model, X_test, y_test)

    st.markdown("---")
    
    # Interactive Prediction
    st.markdown("#### 🎯 Interactive Prediction")
    st.markdown("Enter flight details to predict delay probability:")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    # Get unique values for dropdowns
    # Create airline options with full names
    airline_codes = sorted(flights['AIRLINE'].dropna().unique())
    airline_options = []
    airline_code_map = {}  # Map display name to code
    
    for code in airline_codes:
        airline_info = airlines[airlines['IATA_CODE'] == code]
        if not airline_info.empty:
            airline_name = airline_info.iloc[0]['AIRLINE']
            display_name = f"{code} - {airline_name}"
        else:
            display_name = code
        airline_options.append(display_name)
        airline_code_map[display_name] = code
    
    # Create airport options with full names
    airport_codes = sorted(flights['ORIGIN_AIRPORT'].dropna().unique())
    airport_options = []
    airport_code_map = {}  # Map display name to code
    
    for code in airport_codes:
        airport_info = airports[airports['IATA_CODE'] == code]
        if not airport_info.empty:
            airport_name = airport_info.iloc[0]['AIRPORT']
            city = airport_info.iloc[0]['CITY']
            display_name = f"{code} - {airport_name}, {city}"
        else:
            display_name = code
        airport_options.append(display_name)
        airport_code_map[display_name] = code
    
    with pred_col1:
        input_airline_display = st.selectbox("Airline", airline_options, key='ml_airline')
        input_origin_display = st.selectbox("Origin Airport", airport_options, key='ml_origin')
        input_dest_display = st.selectbox("Destination Airport", airport_options, key='ml_dest')
        
        # Get actual codes from display names
        input_airline = airline_code_map[input_airline_display]
        input_origin = airport_code_map[input_origin_display]
        input_dest = airport_code_map[input_dest_display]
    
    with pred_col2:
        input_month = st.slider("Month", 1, 12, 6, key='ml_month')
        input_dow = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 3, key='ml_dow')
        input_day = st.slider("Day of Month", 1, 31, 15, key='ml_day')
    
    with pred_col3:
        input_sched_dep = st.number_input("Scheduled Departure (24hr format, e.g., 1430)", 
                                          min_value=0, max_value=2359, value=1200, key='ml_sched')
        input_distance = st.number_input("Distance (miles)", 
                                         min_value=0, max_value=5000, value=800, key='ml_dist')
        input_taxi_out = st.slider("Expected Taxi Out Time (min)", 1, 60, 10, key='ml_taxi')
    
    if st.button("Predict Delay Probability", type="primary"):
        prob = predict_delay_probability(
            model, label_encoders, input_airline, input_origin, input_dest,
            input_month, input_dow, input_day, input_sched_dep, input_distance, input_taxi_out
        )
        
        # Store prediction in session state
        st.session_state['prediction_result'] = {
            'prob': prob,
            'airline': input_airline,
            'origin': input_origin,
            'dest': input_dest
        }
    
    # Display prediction results if they exist in session state
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        prob = result['prob']
        
        st.markdown("---")
        st.markdown("#### Prediction Result")
        
        # Display probability as gauge
        prob_pct = prob * 100
        
        # Determine color based on probability
        if prob_pct < 30:
            gauge_color = "#22c55e"  # Green
            risk_level = "Low Risk"
        elif prob_pct < 60:
            gauge_color = "#facc15"  # Yellow
            risk_level = "Medium Risk"
        else:
            gauge_color = "#ef4444"  # Red
            risk_level = "High Risk"
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            fig_prob = create_gauge_chart(prob_pct, "Delay Probability", max_val=100, 
                                         color=gauge_color, unit="%")
            st.plotly_chart(fig_prob, width="stretch")
        
        with result_col2:
            st.markdown(f"""
            <div style="background-color: #1a1c24; border: 2px solid {gauge_color}; border-radius: 8px; padding: 20px; text-align: center; margin-top: 60px;">
                <div style="font-size: 18px; font-weight: bold; color: {gauge_color};">{risk_level}</div>
                <div style="font-size: 14px; color: #94a3b8; margin-top: 10px;">
                    This flight has a <strong>{prob_pct:.1f}%</strong> probability of being delayed.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display Model Performance
    st.markdown("#### 📊 Model Performance")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Accuracy", f"{ml_metrics['accuracy']:.2%}")
    with perf_col2:
        st.metric("Precision", f"{ml_metrics['precision']:.2%}")
    with perf_col3:
        st.metric("Recall", f"{ml_metrics['recall']:.2%}")
    with perf_col4:
        st.metric("F1-Score", f"{ml_metrics['f1']:.2%}")
    
    st.markdown("---")
    
    # Visualizations Row
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = ml_metrics['confusion_matrix']
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: On-Time', 'Predicted: Delayed'],
            y=['Actual: On-Time', 'Actual: Delayed'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=False
        ))
        
        fig_cm.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'family': 'Inter'},
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'},
            height=350
        )
        
        st.plotly_chart(fig_cm, width="stretch", config=PLOTLY_CONFIG)
    
    with viz_col2:
        # Feature Importance
        st.markdown("#### Feature Importance")
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h')
        fig_imp.update_traces(marker_color='#3b82f6')
        fig_imp = update_chart_layout(fig_imp)
        fig_imp.update_layout(height=350, showlegend=False)
        
        st.plotly_chart(fig_imp, width="stretch", config=PLOTLY_CONFIG)

# --- Tab: About ---
with tab_about:
    st.markdown("### The Journey: From Raw Data to Intelligent Predictions")
    
    st.markdown("""
    Every data science project tells a story. This is the story of how we transformed 5.8 million flight records 
    into actionable insights and intelligent predictions that can help both airlines and passengers make better decisions.
    """)
    
    st.markdown("---")
    
    # The Challenge
    st.markdown("#### 🎯 The Challenge")
    st.markdown("""
    Imagine you're planning a trip. You've booked your flight months in advance, packed your bags, and arrived at the airport 
    with plenty of time to spare. Then you see it on the departure board: **DELAYED**. Sound familiar?
    
    Flight delays are more than just an inconvenience—they cost airlines billions of dollars annually and affect millions 
    of passengers. But what if we could predict which flights are likely to be delayed? What if airlines could proactively 
    adjust their schedules, and passengers could make more informed booking decisions?
    
    That's exactly what we set out to do.
    """)
    
    st.markdown("---")
    
    # The Data
    st.markdown("#### 📥 The Data")
    
    st.markdown("""
    Our journey began with a massive dataset from the US Department of Transportation—**5.8 million domestic flight records** 
    from 2015. This wasn't just numbers in a spreadsheet; it was a year's worth of real journeys, real delays, and real 
    stories from travelers across America.
    
    The dataset came in a hefty **500 MB CSV file**, containing everything from departure times and airlines to weather delays 
    and cancellation codes. It was comprehensive, but it was also unwieldy. Loading it took forever, and analyzing it felt 
    like trying to drink from a fire hose.
    """)
    
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        st.info("""
        **📊 The Numbers**  
        5.8 Million flight records  
        500 MB of raw data  
        Full year of 2015  
        Every major US airline
        """)
    
    with data_col2:
        st.info("""
        **🗂️ The Components**  
        Flight schedules & delays  
        Airline information  
        Airport details  
        Cancellation reasons
        """)
    
    st.markdown("---")
    
    # The Transformation
    st.markdown("#### ⚙️ The Transformation")
    
    st.markdown("""
    Here's where things got interesting. We faced our first major challenge: **the data was too slow to work with**. 
    Every time we loaded the CSV file, we'd wait... and wait... and wait some more. This wasn't going to work for an 
    interactive dashboard.
    
    So we got creative. We transformed the data from CSV to **Parquet format**—a columnar storage format designed for 
    analytics. But we didn't stop there. We carefully analyzed which columns we actually needed, dropped the unnecessary 
    ones, and optimized the data types. Think of it like packing for a trip: we kept what we needed and left the rest behind.
    """)
    
    st.code("""
# Before: Slow and memory-hungry
df = pd.read_csv('flights.csv')  # 500 MB, 30+ seconds to load

# After: Fast and efficient
df = pd.read_parquet('flights.parquet')  # 74 MB, loads in seconds!
    """, language="python")
    
    st.markdown("**The result?** A dramatic transformation:")
    
    result_metrics = st.columns(3)
    with result_metrics[0]:
        st.metric("Original Size", "~500 MB", delta=None)
    with result_metrics[1]:
        st.metric("Optimized Size", "74 MB", delta="-426 MB", delta_color="inverse")
    with result_metrics[2]:
        st.metric("Size Reduction", "85%", delta=None)
    
    st.markdown("""
    We didn't just make it smaller—we made it **85% smaller** while keeping all the information we needed. 
    The dashboard now loads in seconds instead of minutes, making exploration feel effortless.
    """)
    
    st.markdown("---")
    
    # The Discovery
    st.markdown("#### 📊 The Discovery")
    
    st.markdown("""
    With our data optimized, we dove into exploration. What patterns were hiding in those 5.8 million flights? 
    
    We discovered fascinating insights: certain airlines consistently outperformed others, some airports were delay hotspots, 
    and delays followed clear temporal patterns—morning flights tended to be more punctual, while evening flights accumulated 
    delays throughout the day like a snowball rolling downhill.
    
    Weather wasn't always the culprit we expected. While it caused its fair share of delays, **airline operations** and 
    **air system issues** were often the real troublemakers. Each visualization in the other tabs tells part of this story, 
    revealing the complex ecosystem of factors that influence whether your flight leaves on time.
    """)
    
    st.markdown("---")
    
    # The Model
    st.markdown("#### 🤖 The Intelligence")
    
    st.markdown("""
    Armed with these insights, we built something powerful: a **machine learning model** that can predict whether a flight 
    will be delayed before it even takes off.
    
    We chose a **Random Forest algorithm**—imagine a committee of 200 decision trees, each analyzing the flight from a 
    different angle, then voting on the outcome. This ensemble approach makes the predictions robust and reliable.
    
    But what does the model actually look at? We engineered **12 carefully selected features** that capture the essence 
    of a flight:
    """)
    
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        st.markdown("""
        **Who & Where**  
        Which airline is operating the flight? Where is it departing from and going to? Different airlines have different 
        track records, and some airport pairs are more challenging than others.
        """)
    
    with model_col2:
        st.markdown("""
        **When & How**  
        What time of day? What day of the week? Is it a weekend? Morning flights behave differently than evening flights. 
        Weekend patterns differ from weekdays.
        """)
    
    st.markdown("""
    **The Details**  
    How far is the flight? How long does taxi-out typically take at this airport? These operational details matter more 
    than you might think—a short taxi time suggests smooth operations, while a long one hints at congestion.
    
    The model learned from millions of examples, discovering patterns that even experienced airline operations managers 
    might miss. It's not perfect—no model is—but it's remarkably good at identifying flights that are at risk of delay.
    """)
    
    st.markdown("---")
    
    # The Validation
    st.markdown("#### ✅ The Validation")
    
    st.markdown("""
    Of course, we couldn't just build a model and call it a day. We needed to know: **does it actually work?**
    
    We held back 20% of our data—over a million flights—that the model had never seen before. This was our test: could 
    the model predict delays for flights it knew nothing about?
    
    The answer was yes. The model demonstrated strong performance across multiple metrics, balancing the need to catch 
    delays (recall) with the need to avoid false alarms (precision). You can see the live metrics in the ML Prediction 
    tab, where the model's confusion matrix tells the full story of its predictions.
    """)
    
    st.markdown("---")
    
    # The Dashboard
    st.markdown("#### 📢 The Dashboard")
    
    st.markdown("""
    Finally, we brought it all together in this interactive dashboard. We didn't want the insights locked away in code 
    or buried in static reports. We wanted them **alive, interactive, and accessible**.
    
    Built with Streamlit and Plotly, this dashboard lets you explore the data yourself, see the patterns we discovered, 
    and even make your own delay predictions. It's designed to be intuitive—no data science degree required—while still 
    providing the depth that analysts and aviation professionals need.
    
    Every chart is interactive. Every metric is real. Every prediction comes from the same model that learned from 
    5.8 million flights. This is data science in action, not just in theory.
    """)
    
    st.markdown("---")
    
    # The Impact
    st.markdown("""
    <div style="background-color: #1a1c24; border-left: 4px solid #3b82f6; padding: 20px; border-radius: 5px; margin-top: 20px;">
        <h4 style="margin-top: 0;">💡 The Impact</h4>
        <p style="font-size: 16px; line-height: 1.6;">
        This project demonstrates what's possible when we combine domain knowledge, technical skill, and creative 
        problem-solving. We took a massive, unwieldy dataset and transformed it into actionable intelligence. We built 
        a model that can help airlines optimize operations and help passengers make smarter travel decisions.
        </p>
        <p style="font-size: 16px; line-height: 1.6; margin-bottom: 0;">
        But more than that, we created a tool that tells a story—the story of millions of flights, thousands of delays, 
        and the patterns that connect them all. Welcome to the journey.
        </p>
    </div>
    """, unsafe_allow_html=True)