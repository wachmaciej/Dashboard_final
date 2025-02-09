import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import datetime
import calendar
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="YOY Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# --- Custom CSS (Optional) ---
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .sidebar .sidebar-content {background-color: #343a40; color: white;}
    /* Reduce gap between columns */
    .css-1lcbmhc.e1tzin5v3 { gap: 0.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Logo ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("YOY Dashboard 📊")
with col2:
    st.image("logo.png", width=300)

# --- Sidebar Uploaders ---
# Main dataset uploader
uploaded_file = st.sidebar.file_uploader("Upload Main Dataset (CSV or Excel)", type=["csv", "xlsx"])
# Advertising data uploader (for Ad Campaign Analysis)
advertising_file = st.sidebar.file_uploader("Upload Advertising Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Please use the file uploader in the sidebar to upload your main dataset and view the dashboard.")
    st.stop()

# =============================================================================
# FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(file):
    """Load main data from CSV or Excel file."""
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    return data

@st.cache_data(show_spinner=False)
def load_ad_data(file):
    """Load advertising data from CSV or Excel file and preprocess it."""
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    
    # Strip extra whitespace from column names
    data.columns = data.columns.str.strip()
    
    # Convert date columns to datetime
    for col in ["Start Date", "End Date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Remove currency symbols and commas, then convert to float
    currency_columns = ["Budget Amount", "Spend", "Last Year Spend", "7 Day Total Sales"]
    for col in currency_columns:
        if col in data.columns:
            data[col] = (data[col]
                         .astype(str)
                         .str.replace(r'[\$,]', '', regex=True)
                         .str.strip()
                         .replace('', '0')
                         .astype(float))
    
    # Process the ACOS column
    if "Total Advertising Cost of Sales (ACOS)" in data.columns:
        data["Total Advertising Cost of Sales (ACOS)"] = (
            data["Total Advertising Cost of Sales (ACOS)"]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
            .replace('', '0')
            .astype(float)
        )
    return data

def get_quarter(week):
    """Convert week number to quarter."""
    if 1 <= week <= 13:
        return "Q1"
    elif 14 <= week <= 26:
        return "Q2"
    elif 27 <= week <= 39:
        return "Q3"
    elif 40 <= week <= 52:
        return "Q4"
    else:
        return None

def format_currency(value):
    """Format number as currency with two decimals (used elsewhere)."""
    return f"£{value:,.2f}"

def format_currency_int(value):
    """Format number as currency without decimals (for KPIs)."""
    return f"£{int(round(value)):,}"

@st.cache_data(show_spinner=False)
def preprocess_data(data):
    """Ensure necessary columns exist and add a 'Quarter' column."""
    required_cols = {"Week", "Year", "Sales Value (£)"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"Dataset is missing one or more required columns: {required_cols}")
        st.stop()
    if "Week" in data.columns:
        data["Quarter"] = data["Week"].apply(get_quarter)
    return data

def week_monday(row):
    """Return the Monday date of a given ISO week."""
    try:
        return datetime.date.fromisocalendar(int(row["Year"]), int(row["Week"]), 1)
    except Exception:
        return None

def get_week_date_range(year, week):
    """Return the Monday and Sunday dates for a given ISO week and year."""
    try:
        monday = datetime.date.fromisocalendar(year, week, 1)
        sunday = datetime.date.fromisocalendar(year, week, 7)
        return monday, sunday
    except Exception:
        return None, None

def create_yoy_trends_chart(data, selected_years, selected_quarters,
                            selected_channels=None, selected_listings=None,
                            selected_products=None):
    """Create a Plotly line chart for YOY Weekly Revenue Trends."""
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
    if selected_listings and len(selected_listings) > 0:
        filtered = filtered[filtered["Listing"].isin(selected_listings)]
    if selected_products and len(selected_products) > 0:
        filtered = filtered[filtered["Product"].isin(selected_products)]
    
    weekly_rev = (filtered.groupby(["Year", "Week"])["Sales Value (£)"]
                  .sum().reset_index().sort_values(by=["Year", "Week"]))
    weekly_rev["RevenueK"] = weekly_rev["Sales Value (£)"] / 1000
    if not filtered.empty:
        min_week, max_week = int(filtered["Week"].min()), int(filtered["Week"].max())
    else:
        min_week, max_week = 1, 52

    fig = px.line(weekly_rev, x="Week", y="Sales Value (£)", color="Year", markers=True,
                  title="Weekly Revenue Trends by Year",
                  labels={"Sales Value (£)": "Revenue (£)"},
                  custom_data=["RevenueK"])
    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K")
    fig.update_layout(xaxis=dict(tickmode="linear", range=[min_week, max_week]),
                      margin=dict(t=50, b=50))
    return fig

def create_pivot_table(data, selected_years, selected_quarters, selected_channels,
                       selected_listings, selected_products, grouping_key="Listing"):
    """
    Create a pivot table of Revenue by Week, applying the desired filters.
    The table is grouped by the provided grouping_key ("Listing" or "Product").
    """
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
    if selected_listings and len(selected_listings) > 0:
        filtered = filtered[filtered["Listing"].isin(selected_listings)]
    if grouping_key == "Product" and selected_products and len(selected_products) > 0:
        filtered = filtered[filtered["Product"].isin(selected_products)]
    
    pivot = pd.pivot_table(filtered, values="Sales Value (£)", index=grouping_key,
                           columns="Week", aggfunc="sum", fill_value=0)
    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0)
    
    # Rename numeric week columns to include a 'Week' prefix.
    new_columns = {}
    for col in pivot.columns:
        if isinstance(col, (int, float)):
            new_columns[col] = f"Week {int(col)}"
    pivot = pivot.rename(columns=new_columns)
    
    return pivot

def create_sku_line_chart(data, sku_text, selected_years, selected_quarter):
    """Create a Plotly line chart for a specific SKU."""
    if "Product SKU" not in data.columns:
        st.error("The dataset does not contain a 'Product SKU' column.")
        st.stop()
    
    filtered = data.copy()
    filtered = filtered[filtered["Product SKU"].str.contains(sku_text, case=False, na=False)]
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarter != "All Quarters":
        filtered = filtered[filtered["Quarter"] == selected_quarter]
    
    if filtered.empty:
        st.warning("No data available for the entered SKU and filters.")
        return None

    weekly_sku = (filtered.groupby(["Year", "Week"])["Sales Value (£)"]
                  .sum().reset_index().sort_values(by=["Year", "Week"]))
    weekly_sku["RevenueK"] = weekly_sku["Sales Value (£)"] / 1000
    if selected_quarter != "All Quarters" and not filtered.empty:
        min_week, max_week = int(filtered["Week"].min()), int(filtered["Week"].max())
    else:
        min_week, max_week = 1, 52

    fig = px.line(weekly_sku, x="Week", y="Sales Value (£)", color="Year", markers=True,
                  title=f"Weekly Revenue Trends for SKU matching: '{sku_text}'",
                  labels={"Sales Value (£)": "Revenue (£)"},
                  custom_data=["RevenueK"])
    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K")
    fig.update_layout(xaxis=dict(tickmode="linear", range=[min_week, max_week]),
                      margin=dict(t=50, b=50))
    return fig

def create_daily_price_chart(data, listing, selected_years, selected_quarters, selected_channels):
    """
    Create a Plotly line chart for the given listing showing average daily price.
    Prices are computed based on the 'Sales Value in Transaction Currency' column divided by 'Order Quantity'.
    The displayed currency is determined by the Original Currency from the filtered data.
    If a Sales Channel filter is provided, the chart will display prices in the corresponding currency;
    if no Sales Channel is selected, it defaults to GBP.
    Missing days are now filled by forward filling the previous recorded daily price.
    Sudden drops and surges are adjusted as per specified rules.
    """
    if "Date" not in data.columns:
        st.error("The dataset does not contain a 'Date' column required for daily price analysis.")
        return None
    
    # Filter by listing, year, quarter, and sales channel
    df_listing = data[(data["Listing"] == listing) & (data["Year"].isin(selected_years))].copy()
    if selected_quarters:
        df_listing = df_listing[df_listing["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        df_listing = df_listing[df_listing["Sales Channel"].isin(selected_channels)]
    
    if df_listing.empty:
        st.warning(f"No data available for {listing} for the selected filters.")
        return None

    # Determine display currency:
    # If a Sales Channel filter is provided, take the Original Currency from the data;
    # otherwise, default to GBP.
    if selected_channels and len(selected_channels) > 0:
        unique_currencies = df_listing["Original Currency"].unique()
        display_currency = unique_currencies[0] if len(unique_currencies) > 0 else "GBP"
    else:
        display_currency = "GBP"

    df_listing["Date"] = pd.to_datetime(df_listing["Date"])
    # Use the Sales Value in Transaction Currency column for calculations
    grouped = df_listing.groupby([df_listing["Date"].dt.date, "Year"]).agg({
        "Sales Value in Transaction Currency": "sum",
        "Order Quantity": "sum"
    }).reset_index()
    grouped.rename(columns={"Date": "Date"}, inplace=True)
    grouped["Average Price"] = grouped["Sales Value in Transaction Currency"] / grouped["Order Quantity"]
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    
    dfs = []
    for yr in selected_years:
        df_year = grouped[grouped["Year"] == yr].copy()
        if df_year.empty:
            continue
        df_year["Day"] = df_year["Date"].dt.dayofyear
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())
        # Reindex so all days in the range are present
        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))
        df_year.index.name = "Day"
        # Instead of linear interpolation, forward fill the missing daily price values.
        df_year["Average Price"] = df_year["Average Price"].ffill()
        
        # --- Adjustment: Apply rules for sudden drops and surges ---
        prices = df_year["Average Price"].values.copy()
        for i in range(1, len(prices)):
            if prices[i] < 0.75 * prices[i-1]:
                prices[i] = prices[i-1]
            if prices[i] > 1.25 * prices[i-1]:
                prices[i] = prices[i-1]
        df_year["Average Price"] = prices
        
        df_year["Year"] = yr
        df_year = df_year.reset_index()
        dfs.append(df_year)
    
    if not dfs:
        st.warning("No data available after processing for the selected filters.")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    fig = px.line(
        combined, 
        x="Day", 
        y="Average Price", 
        color="Year",
        title=f"Daily Average Price for {listing}",
        labels={
            "Day": "Day of Year", 
            "Average Price": f"Average Price ({display_currency})"
        },
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50))
    return fig

# =============================================================================
# MAIN CODE
# =============================================================================

df = load_data(uploaded_file)
df = preprocess_data(df)

available_years = sorted(df["Year"].dropna().unique())
if not available_years:
    st.error("No year data available.")
    st.stop()
current_year = available_years[-1]
if len(available_years) >= 2:
    prev_year = available_years[-2]
    yoy_default_years = [prev_year, current_year]
else:
    yoy_default_years = [current_year]
default_current_year = [current_year]

# Reorganize tabs:
tabs = st.tabs([
    "KPIs", 
    "YOY Trends", 
    "Daily Prices", 
    "SKU Trends", 
    "Pivot Table", 
    "Ad Campaign Analysis",
    "Unrecognised Sales"
])

# -----------------------------------------
# Tab 1: KPIs (Custom HTML Display for Full Control)
# -----------------------------------------
with tabs[0]:
    st.header("Key Performance Indicators")
    with st.expander("KPI Filters", expanded=False):
        today = datetime.date.today()
        last_full_week_end = today - datetime.timedelta(days=today.weekday() + 1) if today.weekday() != 6 else today
        default_week = last_full_week_end.isocalendar()[1]
        available_weeks = sorted(df["Week"].dropna().unique())
        default_index = available_weeks.index(default_week) if default_week in available_weeks else 0
        selected_week = st.selectbox("Select Week for KPI Calculation",
                                     options=available_weeks,
                                     index=default_index,
                                     key="kpi_week",
                                     help="Select the week to calculate KPIs for. (Defaults to the last full week)")
        monday, sunday = get_week_date_range(current_year, selected_week)
        if monday and sunday:
            st.info(f"Wk {selected_week}: {monday.strftime('%d %b')} - {sunday.strftime('%d %b, %Y')}")
    
    # Compute KPI data
    kpi_data = df[df["Week"] == selected_week]
    kpi_summary = kpi_data.groupby("Year")["Sales Value (£)"].sum()
    all_years = sorted(df["Year"].dropna().unique())
    kpi_summary = kpi_summary.reindex(all_years, fill_value=0)
    
    # Create columns for each year
    kpi_cols = st.columns(len(all_years))
    for idx, year in enumerate(all_years):
        with kpi_cols[idx]:
            # Display the year header
            st.subheader(f"Year {year}")
            # Display week indication directly under the year header
            st.markdown(f"<div style='font-size: 1.2em;'>Week {selected_week} Revenue</div>", unsafe_allow_html=True)
            if kpi_summary[year] == 0:
                st.write("Revenue: N/A")
            else:
                revenue_str = format_currency_int(kpi_summary[year])
                # For years beyond the first, calculate delta (difference with previous year)
                if idx > 0:
                    delta_val = int(round(kpi_summary[year] - kpi_summary[all_years[idx - 1]]))
                    if delta_val > 0:
                        # For positive values, display upward arrow (green)
                        delta_display = f"↑ {delta_val:,}"
                        color = "green"
                    elif delta_val < 0:
                        # For negative values, display downward arrow with minus sign (red)
                        delta_display = f"↓ -{abs(delta_val):,}"
                        color = "red"
                    else:
                        delta_display = ""
                        color = "black"
                else:
                    # For the first year, leave delta blank.
                    delta_display = ""
                    color = "black"
                # Display the revenue and delta using custom HTML
                st.markdown(f"<div style='font-size: 1.5em;'>{revenue_str}</div>", unsafe_allow_html=True)
                if delta_display:
                    st.markdown(f"<div style='font-size: 1.2em; color: {color};'>{delta_display}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(f"Top Gainers and Losers for Week {selected_week} ({current_year})")
    
    # Top Gainers and Losers (with custom formatting)
    current_year_data = df[df["Year"] == current_year]
    week_data_current = current_year_data[current_year_data["Week"] == selected_week]
    if selected_week > 1:
        previous_week = selected_week - 1
        week_data_prev = current_year_data[current_year_data["Week"] == previous_week]
    else:
        previous_years = [year for year in available_years if year < current_year]
        if previous_years:
            prev_year = max(previous_years)
            week_data_prev = df[(df["Year"] == prev_year) & (df["Week"] == 52)]
        else:
            week_data_prev = pd.DataFrame(columns=current_year_data.columns)
    
    rev_current = week_data_current.groupby("Listing")["Sales Value (£)"].sum()
    rev_previous = week_data_prev.groupby("Listing")["Sales Value (£)"].sum()
    combined = pd.concat([rev_current, rev_previous], axis=1, keys=["Current", "Previous"]).fillna(0)
    
    def compute_abs_change(row):
        return None if row["Previous"] == 0 else (row["Current"] - row["Previous"])
    
    combined["abs_change"] = combined.apply(compute_abs_change, axis=1)
    num_items = 3
    top_gainers = combined[combined["abs_change"].notnull()].sort_values("abs_change", ascending=False).head(num_items)
    top_losers = combined[combined["abs_change"].notnull()].sort_values("abs_change", ascending=True).head(num_items)
    
    def format_abs_change(val):
        if val is None:
            return "<span style='color:gray;'>N/A</span>"
        if val > 0:
            return f"<span style='color:green;'>↑ {int(round(val)):,}</span>"
        elif val < 0:
            return f"<span style='color:red;'>↓ -{int(round(abs(val))):,}</span>"
        else:
            return f"<span style='color:gray;'>→ {int(round(val)):,}</span>"
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div style='text-align:center;'><h4>Top Gainers</h4></div>", unsafe_allow_html=True)
        if top_gainers.empty:
            st.info("No gainers data available for this week.")
        else:
            for listing, row in top_gainers.iterrows():
                change_html = format_abs_change(row["abs_change"])
                st.markdown(f"""
                    <div style="background:#94f7bb; color:black; padding:10px; border-radius:8px; margin-bottom:8px; text-align:center;">
                        <strong>{listing}</strong><br>
                        {change_html}<br>{format_currency_int(row['Current'])}
                    </div>
                """, unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align:center;'><h4>Top Losers</h4></div>", unsafe_allow_html=True)
        if top_losers.empty:
            st.info("No losers data available for this week.")
        else:
            for listing, row in top_losers.iterrows():
                change_html = format_abs_change(row["abs_change"])
                st.markdown(f"""
                    <div style="background:#fcb8b8; color:black; padding:10px; border-radius:8px; margin-bottom:8px; text-align:center;">
                        <strong>{listing}</strong><br>
                        {change_html}<br>{format_currency_int(row['Current'])}
                    </div>
                """, unsafe_allow_html=True)

# -----------------------------------------
# Tab 2: YOY Trends
# -----------------------------------------
with tabs[1]:
    st.header("YOY Weekly Revenue Trends")
    with st.expander("Chart Filters", expanded=False):
        yoy_years = st.multiselect("Select Year(s)", options=available_years,
                                   default=yoy_default_years, key="yoy_years",
                                   help="Default is previous and current year.")
        selected_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"],
                                           default=["Q1", "Q2", "Q3", "Q4"], key="yoy_quarters",
                                           help="Select one or more quarters to filter by.")
        selected_channels = st.multiselect("Select Sales Channel(s)",
                                           options=sorted(df["Sales Channel"].dropna().unique()),
                                           default=[], key="yoy_channels",
                                           help="Select one or more channels to filter. If empty, all channels are shown.")
        selected_listings = st.multiselect("Select Listing(s)",
                                           options=sorted(df["Listing"].dropna().unique()),
                                           default=[], key="yoy_listings",
                                           help="Select one or more listings to filter.")
        if selected_listings:
            product_options = sorted(df[df["Listing"].isin(selected_listings)]["Product"].dropna().unique())
        else:
            product_options = sorted(df["Product"].dropna().unique())
        selected_products = st.multiselect("Select Product(s)", options=product_options,
                                           default=[], key="yoy_products",
                                           help="Select one or more products to filter (affects the line chart only).")
    fig_yoy = create_yoy_trends_chart(df, yoy_years, selected_quarters, selected_channels, selected_listings, selected_products)
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown("### Revenue Summary")
    st.markdown("")
    filtered_df = df.copy()
    if yoy_years:
        filtered_df = filtered_df[filtered_df["Year"].isin(yoy_years)]
    if selected_quarters:
        filtered_df = filtered_df[filtered_df["Quarter"].isin(selected_quarters)]
    if selected_channels:
        filtered_df = filtered_df[filtered_df["Sales Channel"].isin(selected_channels)]
    if selected_listings:
        filtered_df = filtered_df[filtered_df["Listing"].isin(selected_listings)]
    df_revenue = filtered_df.copy()
    if df_revenue.empty:
        st.info("No data available for the selected filters to build the revenue summary table.")
    else:
        df_revenue["Year"] = df_revenue["Year"].astype(int)
        df_revenue["Week"] = df_revenue["Week"].astype(int)
        filtered_current_year = df_revenue["Year"].max()
        df_revenue_current = df_revenue[df_revenue["Year"] == filtered_current_year].copy()
        df_revenue_current["Week_Monday"] = df_revenue_current.apply(week_monday, axis=1)
        df_revenue_current["Week_Sunday"] = df_revenue_current["Week_Monday"].apply(lambda d: d + datetime.timedelta(days=6) if d else None)
        today = datetime.date.today()
        last_complete_week_end = today - datetime.timedelta(days=today.weekday() + 1) if today.weekday() != 6 else today
        df_full_weeks_current = df_revenue_current[df_revenue_current["Week_Sunday"] <= last_complete_week_end].copy()
        unique_weeks_current = (df_full_weeks_current.groupby(["Year", "Week"])
                                .first()
                                .reset_index()[["Year", "Week", "Week_Sunday"]]
                                .sort_values("Week_Sunday"))
        if unique_weeks_current.empty:
            st.info("Not enough complete week data in the filtered current year to build the revenue summary table.")
        else:
            last_complete_week_row_current = unique_weeks_current.iloc[-1]
            last_week_tuple_current = (last_complete_week_row_current["Year"], last_complete_week_row_current["Week"])
            last_4_weeks_current = unique_weeks_current.tail(4)
            last_4_week_tuples_current = set(last_4_weeks_current[["Year", "Week"]].apply(tuple, axis=1))
            grouping_key = "Product" if selected_listings and len(selected_listings) == 1 else "Listing"
            rev_last_4_current = (df_full_weeks_current[df_full_weeks_current.apply(lambda row: (row["Year"], row["Week"]) in last_4_week_tuples_current, axis=1)]
                                   .groupby(grouping_key)["Sales Value (£)"]
                                   .sum()
                                   .rename("Last 4 Weeks Revenue (Current Year)")
                                   .round(0).astype(int))
            rev_last_1_current = (df_full_weeks_current[df_full_weeks_current.apply(lambda row: (row["Year"], row["Week"]) == last_week_tuple_current, axis=1)]
                                   .groupby(grouping_key)["Sales Value (£)"]
                                   .sum()
                                   .rename("Last Week Revenue (Current Year)")
                                   .round(0).astype(int))
            if len(filtered_df["Year"].unique()) >= 2:
                filtered_years = sorted(filtered_df["Year"].unique())
                last_year = filtered_years[-2]
                reference_date_last_year = datetime.date(last_year, 12, 31)
                last_complete_week_end_last_year = (reference_date_last_year -
                                                    datetime.timedelta(days=reference_date_last_year.weekday() + 1)
                                                    if reference_date_last_year.weekday() != 6 else reference_date_last_year)
                df_revenue_last_year = df_revenue[df_revenue["Year"] == last_year].copy()
                df_revenue_last_year["Week_Monday"] = df_revenue_last_year.apply(week_monday, axis=1)
                df_revenue_last_year["Week_Sunday"] = df_revenue_last_year["Week_Monday"].apply(lambda d: d + datetime.timedelta(days=6) if d else None)
                df_full_weeks_last_year = df_revenue_last_year[df_revenue_last_year["Week_Sunday"] <= last_complete_week_end_last_year].copy()
                unique_weeks_last_year = (df_full_weeks_last_year.groupby(["Year", "Week"])
                                          .first()
                                          .reset_index()[["Year", "Week", "Week_Sunday"]]
                                          .sort_values("Week_Sunday"))
                if unique_weeks_last_year.empty:
                    rev_last_4_last_year = pd.Series(dtype=float)
                    rev_last_1_last_year = pd.Series(dtype=float)
                else:
                    last_complete_week_row_last_year = unique_weeks_last_year.iloc[-1]
                    last_week_tuple_last_year = (last_complete_week_row_last_year["Year"], last_complete_week_row_last_year["Week"])
                    last_4_weeks_last_year = unique_weeks_last_year.tail(4)
                    last_4_week_tuples_last_year = set(last_4_weeks_last_year[["Year", "Week"]].apply(tuple, axis=1))
                    rev_last_4_last_year = (df_full_weeks_last_year[df_full_weeks_last_year.apply(lambda row: (row["Year"], row["Week"]) in last_4_week_tuples_last_year, axis=1)]
                                             .groupby(grouping_key)["Sales Value (£)"]
                                             .sum()
                                             .rename("Last 4 Weeks Revenue (Last Year)")
                                             .round(0).astype(int))
                    rev_last_1_last_year = (df_full_weeks_last_year[df_full_weeks_last_year.apply(lambda row: (row["Year"], row["Week"]) == last_week_tuple_last_year, axis=1)]
                                             .groupby(grouping_key)["Sales Value (£)"]
                                             .sum()
                                             .rename("Last Week Revenue (Last Year)")
                                             .round(0).astype(int))
            else:
                rev_last_4_last_year = pd.Series(dtype=float)
                rev_last_1_last_year = pd.Series(dtype=float)
            all_keys_current = pd.Series(sorted(df_revenue_current[grouping_key].unique()), name=grouping_key)
            revenue_summary = pd.DataFrame(all_keys_current).set_index(grouping_key)
            revenue_summary = revenue_summary.join(rev_last_4_current, how="left").join(rev_last_1_current, how="left")
            revenue_summary = revenue_summary.fillna(0)
            revenue_summary = revenue_summary.join(rev_last_4_last_year, how="left").join(rev_last_1_last_year, how="left")
            revenue_summary = revenue_summary.fillna(0).reset_index()
            for col in ["Last 4 Weeks Revenue (Current Year)", "Last Week Revenue (Current Year)",
                        "Last 4 Weeks Revenue (Last Year)", "Last Week Revenue (Last Year)"]:
                revenue_summary[col] = revenue_summary[col].astype(int)
            def format_diff_display(row, current_col, last_year_col):
                curr = row[current_col]
                prev = row[last_year_col]
                diff = curr - prev
                if prev == 0:
                    return f"{int(diff):,} (N/A)"
                else:
                    pct = diff / prev * 100
                    return f"{int(diff):,}      ({pct:+.0f}%)"
            revenue_summary["Last 4 Weeks Diff"] = revenue_summary.apply(lambda row: format_diff_display(row, "Last 4 Weeks Revenue (Current Year)", "Last 4 Weeks Revenue (Last Year)"), axis=1)
            revenue_summary["Last Week Diff"] = revenue_summary.apply(lambda row: format_diff_display(row, "Last Week Revenue (Current Year)", "Last Week Revenue (Last Year)"), axis=1)
            # Change desired order: move "Last 4 Weeks Diff" to be positioned after "Last 4 Weeks Revenue (Last Year)"
            desired_order = [grouping_key,
                             "Last 4 Weeks Revenue (Current Year)",
                             "Last 4 Weeks Revenue (Last Year)",
                             "Last 4 Weeks Diff",
                             "Last Week Revenue (Current Year)",
                             "Last Week Revenue (Last Year)",
                             "Last Week Diff"]
            revenue_summary = revenue_summary[desired_order]
            
            total_last_4_current = revenue_summary["Last 4 Weeks Revenue (Current Year)"].sum()
            total_last_4_last_year = revenue_summary["Last 4 Weeks Revenue (Last Year)"].sum()
            total_last_week_current = revenue_summary["Last Week Revenue (Current Year)"].sum()
            total_last_week_last_year = revenue_summary["Last Week Revenue (Last Year)"].sum()
            
            diff_4 = total_last_4_current - total_last_4_last_year
            if total_last_4_last_year == 0:
                diff_4_str = f"{diff_4:,} (N/A)"
            else:
                pct_4 = diff_4 / total_last_4_last_year * 100
                diff_4_str = f"{diff_4:,}      ({pct_4:+.0f}%)"
            
            diff_week = total_last_week_current - total_last_week_last_year
            if total_last_week_last_year == 0:
                diff_week_str = f"{diff_week:,} (N/A)"
            else:
                pct_week = diff_week / total_last_week_last_year * 100
                diff_week_str = f"{diff_week:,}      ({pct_week:+.0f}%)"
            
            summary_row = {
                grouping_key: "Total",
                "Last 4 Weeks Revenue (Current Year)": total_last_4_current,
                "Last 4 Weeks Revenue (Last Year)": total_last_4_last_year,
                "Last Week Revenue (Current Year)": total_last_week_current,
                "Last Week Revenue (Last Year)": total_last_week_last_year,
                "Last 4 Weeks Diff": diff_4_str,
                "Last Week Diff": diff_week_str
            }
            
            revenue_summary = pd.concat([revenue_summary, pd.DataFrame([summary_row])], ignore_index=True)
            
            st.markdown("Click on any column header to sort")
            def color_diff(val):
                try:
                    if "N/A" in val:
                        return ""
                    start = val.index('(') + 1
                    end = val.index(')')
                    pct_text = val[start:end].strip()
                    pct_value = float(pct_text.replace('%', '').replace('+', '').strip())
                    return 'color: green' if pct_value > 0 else 'color: red' if pct_value < 0 else ''
                except Exception:
                    return ''
            
            def highlight_total_row(row):
                if row[grouping_key] == "Total":
                    return ['font-weight: bold'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_revenue_summary = (
                revenue_summary.style
                .format({
                    "Last 4 Weeks Revenue (Current Year)": "{:,}",
                    "Last 4 Weeks Revenue (Last Year)": "{:,}",
                    "Last Week Revenue (Current Year)": "{:,}",
                    "Last Week Revenue (Last Year)": "{:,}"
                })
                .apply(highlight_total_row, axis=1)
                .applymap(color_diff, subset=["Last 4 Weeks Diff", "Last Week Diff"])
            )
            
            st.dataframe(styled_revenue_summary, use_container_width=True)

# -----------------------------------------
# Tab 3: Daily Prices
# -----------------------------------------
with tabs[2]:
    st.header("Daily Prices for Top Listings")
    with st.expander("Daily Price Filters", expanded=False):
        default_daily_years = [year for year in available_years if year in (2024, 2025)]
        if not default_daily_years:
            default_daily_years = [current_year]
        selected_daily_years = st.multiselect("Select Year(s) to compare", options=available_years,
                                              default=default_daily_years, key="daily_years",
                                              help="Default shows 2024 and 2025 if available.")
        selected_daily_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"],
                                                 default=["Q1", "Q2", "Q3", "Q4"], key="daily_quarters",
                                                 help="Select one or more quarters to filter.")
        selected_daily_channels = st.multiselect("Select Sales Channel(s)",
                                                 options=sorted(df["Sales Channel"].dropna().unique()),
                                                 default=[], key="daily_channels",
                                                 help="Select one or more sales channels to filter the daily price data.")
    main_listings = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Patterned Polos"]
    for listing in main_listings:
        st.subheader(listing)
        fig_daily = create_daily_price_chart(df, listing, selected_daily_years, selected_daily_quarters, selected_daily_channels)
        if fig_daily:
            st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("## Daily Prices Comparison")
    with st.expander("Comparison Chart Filters", expanded=False):
        comp_years = st.multiselect("Select Year(s)", options=available_years,
                                    default=default_daily_years, key="comp_years",
                                    help="Select the year(s) for the comparison chart.")
        comp_channels = st.multiselect("Select Sales Channel(s)",
                                       options=sorted(df["Sales Channel"].dropna().unique()),
                                       default=[], key="comp_channels",
                                       help="Select the sales channel(s) for the comparison chart.")
        comp_listing = st.selectbox("Select Listing", options=sorted(df["Listing"].dropna().unique()),
                                    key="comp_listing",
                                    help="Select a listing for daily prices comparison.")
    all_quarters = ["Q1", "Q2", "Q3", "Q4"]
    fig_comp = create_daily_price_chart(df, comp_listing, comp_years, all_quarters, comp_channels)
    if fig_comp:
        st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------------------------
# Tab 4: SKU Trends
# -----------------------------------------
with tabs[3]:
    st.header("SKU Trends")
    if "Product SKU" not in df.columns:
        st.error("The dataset does not contain a 'Product SKU' column.")
    else:
        with st.expander("SKU Chart Filters", expanded=True):
            sku_text = st.text_input("Enter Product SKU", value="",
                                     key="sku_input",
                                     help="Enter a SKU (or part of it) to display its weekly revenue trends.")
            sku_years = st.multiselect("Select Year(s)", options=available_years,
                                       default=default_current_year, key="sku_years",
                                       help="Default is the current year.")
            sku_quarter = st.selectbox("Select Quarter",
                                       options=["All Quarters", "Q1", "Q2", "Q3", "Q4"],
                                       index=0, key="sku_quarter",
                                       help="Select a specific quarter or All Quarters.")
        if sku_text.strip() == "":
            st.info("Please enter a Product SKU to view its trends.")
        else:
            fig_sku = create_sku_line_chart(df, sku_text, sku_years, sku_quarter)
            if fig_sku is not None:
                st.plotly_chart(fig_sku, use_container_width=True)

# -----------------------------------------
# Tab 5: Pivot Table
# -----------------------------------------
with tabs[4]:
    st.header("Pivot Table: Revenue by Week")
    with st.expander("Pivot Table Filters", expanded=False):
        pivot_years = st.multiselect("Select Year(s) for Pivot Table", options=available_years,
                                     default=default_current_year, key="pivot_years",
                                     help="Default is the current year.")
        pivot_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"],
                                        default=["Q1", "Q2", "Q3", "Q4"], key="pivot_quarters",
                                        help="Select one or more quarters to filter by.")
        pivot_channels = st.multiselect("Select Sales Channel(s)",
                                        options=sorted(df["Sales Channel"].dropna().unique()),
                                        default=[], key="pivot_channels",
                                        help="Select one or more channels to filter. If empty, all channels are shown.")
        pivot_listings = st.multiselect("Select Listing(s)",
                                        options=sorted(df["Listing"].dropna().unique()),
                                        default=[], key="pivot_listings",
                                        help="Select one or more listings to filter. If empty, all listings are shown.")
        if pivot_listings and len(pivot_listings) == 1:
            pivot_product_options = sorted(df[df["Listing"] == pivot_listings[0]]["Product"].dropna().unique())
        else:
            pivot_product_options = sorted(df["Product"].dropna().unique())
        pivot_products = st.multiselect("Select Product(s)",
                                        options=pivot_product_options,
                                        default=[], key="pivot_products",
                                        help="Select one or more products to filter (only applies if a specific listing is selected).")
    grouping_key = "Product" if (pivot_listings and len(pivot_listings) == 1) else "Listing"
    effective_products = pivot_products if grouping_key == "Product" else []
    pivot = create_pivot_table(df,
                               selected_years=pivot_years,
                               selected_quarters=pivot_quarters,
                               selected_channels=pivot_channels,
                               selected_listings=pivot_listings,
                               selected_products=effective_products,
                               grouping_key=grouping_key)
    if len(pivot_years) == 1:
        year_for_date = int(pivot_years[0])
        new_columns = []
        for col in pivot.columns:
            if col == "Total Revenue":
                new_columns.append((col, "Total Revenue"))
            else:
                try:
                    week_number = int(col.split()[1])
                    mon, sun = get_week_date_range(year_for_date, week_number)
                    date_range = f"{mon.strftime('%d %b')} - {sun.strftime('%d %b')}" if mon and sun else ""
                    new_columns.append((col, date_range))
                except Exception:
                    new_columns.append((col, ""))
        pivot.columns = pd.MultiIndex.from_tuples(new_columns)
    st.dataframe(pivot, use_container_width=True)

# -----------------------------------------
# Tab 6: Ad Campaign Analysis
# -----------------------------------------
with tabs[5]:
    st.header("Ad Campaign Analysis")
    if advertising_file is None:
        st.info("Please upload an advertising dataset to view the analysis in this tab.")
    else:
        ad_data = load_ad_data(advertising_file)
        st.markdown("## Campaign Comparison")
        with st.expander("Compare Monthly Metrics", expanded=False):
            if "Start Date" in ad_data.columns:
                ad_data["Start Date"] = pd.to_datetime(ad_data["Start Date"], errors='coerce')
                ad_data["Month"] = ad_data["Start Date"].dt.strftime("%B")
                ad_data["Year"] = ad_data["Start Date"].dt.year
                available_months = sorted(ad_data["Month"].dropna().unique(), key=lambda m: list(calendar.month_name).index(m))
                available_years_ad = sorted(ad_data["Year"].dropna().unique())
            else:
                available_months = list(calendar.month_name)[1:]
                available_years_ad = []
            selected_month = st.selectbox("Select Month", options=available_months, index=0)
            selected_ad_years = st.multiselect("Select Year(s)", options=available_years_ad, default=available_years_ad)
            available_listings_ad = sorted(ad_data["Portfolio name"].dropna().unique())
            selected_listing = st.selectbox("Select Listing", options=available_listings_ad, index=0)
            filtered = ad_data[(ad_data["Month"] == selected_month) &
                               (ad_data["Year"].isin(selected_ad_years)) &
                               (ad_data["Portfolio name"] == selected_listing)]
            if filtered.empty:
                st.warning("No advertising data available for the selected month, years, and listing filter.")
            else:
                agg = filtered.groupby(["Campaign Name", "Year"]).agg({
                    "7 Day Total Sales": "sum",
                    "Spend": "sum"
                }).reset_index()
                agg.rename(columns={"7 Day Total Sales": "Revenue"}, inplace=True)
                agg["ACOS"] = agg.apply(lambda row: (row["Spend"] / row["Revenue"] * 100) if row["Revenue"] > 0 else 0, axis=1)
                category_orders = {"Year": sorted(filtered["Year"].unique())}
                rev_fig = px.bar(agg, x="Campaign Name", y="Revenue", color="Year",
                                 barmode="group", category_orders=category_orders,
                                 title=f"Revenue for {selected_month} - {selected_listing}",
                                 labels={"Revenue": "Revenue (£)", "Campaign Name": "Campaign"})
                spend_fig = px.bar(agg, x="Campaign Name", y="Spend", color="Year",
                                   barmode="group", category_orders=category_orders,
                                   title=f"Spend for {selected_month} - {selected_listing}",
                                   labels={"Spend": "Spend (£)", "Campaign Name": "Campaign"})
                acos_fig = px.bar(agg, x="Campaign Name", y="ACOS", color="Year",
                                  barmode="group", category_orders=category_orders,
                                  title=f"ACOS (%) for {selected_month} - {selected_listing}",
                                  labels={"ACOS": "ACOS (%)", "Campaign Name": "Campaign"})
                st.plotly_chart(rev_fig, use_container_width=True)
                st.plotly_chart(spend_fig, use_container_width=True)
                st.plotly_chart(acos_fig, use_container_width=True)
                st.markdown("### Available Campaigns for Selected Listing")
                display_cols = ["Campaign Name", "Portfolio name", "Program Type", "Status", "Spend", "7 Day Total Sales", "Total Advertising Cost of Sales (ACOS)"]
                st.dataframe(filtered[display_cols])

# -----------------------------------------
# Tab 7: Unrecognised Sales
# -----------------------------------------
with tabs[6]:
    st.header("Unrecognised Sales")
    # Filter rows where the Listing column contains "Unrecognised" (case-insensitive)
    unrecognised_sales = df[df["Listing"].str.contains("unrecognised", case=False, na=False)]
    # Drop the specified columns
    columns_to_drop = ["Year", "Weekly Sales Value (£)", "YOY Growth (%)"]
    unrecognised_sales = unrecognised_sales.drop(columns=columns_to_drop, errors='ignore')
    if unrecognised_sales.empty:
        st.info("No unrecognised sales found.")
    else:
        st.dataframe(unrecognised_sales, use_container_width=True)
