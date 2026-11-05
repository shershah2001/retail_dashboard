import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Retail Business Intelligence", page_icon="📊", layout="wide")

NAVY = "#142B4A"
BLUE = "#2F6B9A"
GREEN = "#238B57"
RED = "#C64B4B"
BG = "#F4F7FB"

st.markdown("""
<style>
.stApp {background: #F4F7FB;}
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
h1,h2,h3 {color:#142B4A;}
[data-testid="stMetric"] {background:white; border:1px solid #E5EAF1; padding:14px 16px; border-radius:12px;}
[data-testid="stSidebar"] {background:#FFFFFF;}
div[data-testid="stTabs"] button {font-weight:650;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading cleaned retail dataset...")
def load_data(path):
    d = pd.read_csv(path, low_memory=False)

    # Normalize column names
    d.columns = d.columns.str.strip()

    # Support both original and cleaned date column names
    if "order_date" not in d.columns and "InvoiceDate" in d.columns:
        d.rename(columns={"InvoiceDate": "order_date"}, inplace=True)

    # Validate important columns
    required_columns = [
        "order_date",
        "transaction_id",
        "total_amount",
        "customer_id",
        "quantity",
        "unit_price"
    ]

    missing = [col for col in required_columns if col not in d.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {d.columns.tolist()}"
        )

    # Convert dates
    d["order_date"] = pd.to_datetime(
        d["order_date"], errors="coerce"
    )

    if "delivery_date" in d.columns:
        d["delivery_date"] = pd.to_datetime(
            d["delivery_date"], errors="coerce"
        )
    else:
        d["delivery_date"] = pd.NaT

    # Convert numeric columns
    numeric_columns = [
        "quantity", "unit_price", "discount_pct",
        "total_amount", "age", "rating"
    ]

    for col in numeric_columns:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Date-based calculated columns
    d["order_year"] = d["order_date"].dt.year
    d["month_num"] = d["order_date"].dt.month
    d["month_label"] = d["order_date"].dt.strftime("%b")

    d["delivery_days"] = (
        d["delivery_date"] - d["order_date"]
    ).dt.days

    d["weekday"] = d["order_date"].dt.day_name()

    d["day_type_calc"] = np.where(
        d["order_date"].dt.dayofweek >= 5,
        "Weekend",
        "Weekday"
    )

    return d


with st.sidebar:
    st.header("Dashboard Controls")

    data_path = "https://huggingface.co/datasets/welacoder2001/retail-sales-dashboard/resolve/main/final_cleaned_data.csv"

    try:
        raw = load_data(data_path)

        st.success("Dataset loaded successfully!")

        # Temporary debugging information
        st.write("Dataset shape:", raw.shape)
        st.write("Available columns:", raw.columns.tolist())

    except Exception as e:
        st.error(f"Could not load the CSV: {e}")
        st.stop()

    valid_dates = raw["order_date"].dropna()

    if valid_dates.empty:
        st.error("No valid order_date values found.")
        st.stop()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    date_range = st.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = pd.Timestamp(min_date)
        end_date = pd.Timestamp(max_date)

    def options(col):
        return (
            sorted(raw[col].dropna().astype(str).unique().tolist())
            if col in raw else []
        )

    states = st.multiselect("State", options("state"))
    cities = st.multiselect("City", options("city"))
    categories = st.multiselect("Category", options("category"))
    brands = st.multiselect("Brand", options("brand"))
    genders = st.multiselect("Gender", options("gender"))
    statuses = st.multiselect("Order status", options("order_status"))
    payments = st.multiselect("Payment method", options("payment"))


st.title("Retail Sales & Business Intelligence")
st.caption("Interactive portfolio dashboard · Based on the supplied cleaned retail dataset")

filtered = raw.loc[raw["order_date"].between(start_date, end_date)].copy()
for col, selected in [("state", states), ("city", cities), ("category", categories),
                      ("brand", brands), ("gender", genders), ("order_status", statuses),
                      ("payment", payments)]:
    if selected:
        filtered = filtered[filtered[col].astype(str).isin(selected)]

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

def money(v):
    if pd.isna(v): return "—"
    v = float(v)
    a = abs(v)
    if a >= 1e7: return f"₹{v/1e7:,.2f} Cr"
    if a >= 1e5: return f"₹{v/1e5:,.2f} L"
    return f"₹{v:,.0f}"

def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def revenue_data(d):
    return d["total_amount"].fillna(0)

def delivered(d):
    return d[d["order_status"].astype(str).str.lower().eq("delivered")]

def monthly_revenue(d):
    x = d.dropna(subset=["order_date"]).set_index("order_date")["total_amount"].fillna(0).resample("MS").sum()
    return x



def render_business_insights(title, findings, interpretations, actions):
    """Render concise, data-driven findings, interpretations and actions."""
    st.markdown("---")
    st.subheader(f"💡 {title}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### A. Key Findings")
        for item in findings:
            st.markdown(f"- {item}")
    with col2:
        st.markdown("#### B. Business Interpretation")
        for item in interpretations:
            st.markdown(f"- {item}")
    with col3:
        st.markdown("#### C. Recommended Actions")
        for item in actions:
            st.markdown(f"- {item}")


def safe_pct(part, whole):
    return (part / whole * 100) if pd.notna(whole) and whole else np.nan


def signed_pct(value):
    return f"{value:+.1f}%" if pd.notna(value) else "not available"


def get_monthly_change(monthly):
    monthly = monthly.dropna()
    if len(monthly) < 2 or monthly.iloc[-2] == 0:
        return np.nan, None, None
    change = (monthly.iloc[-1] - monthly.iloc[-2]) / abs(monthly.iloc[-2]) * 100
    return change, monthly.index[-2], monthly.index[-1]



def chart(fig, height=350):
    fig.update_layout(
        height=height, template="plotly_white", margin=dict(l=15, r=15, t=55, b=15),
        title_font=dict(color=NAVY, size=17), font=dict(color="#334155"),
        paper_bgcolor="white", plot_bgcolor="white", legend_title_text=""
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

orders = filtered["transaction_id"].nunique()
customers = filtered["customer_id"].nunique()
total_rev = revenue_data(filtered).sum()
deliv = delivered(filtered)
deliv_orders = deliv["transaction_id"].nunique()
deliv_rev = revenue_data(deliv).sum()
aov = deliv_rev / deliv_orders if deliv_orders else np.nan

tabs = st.tabs([
    "Executive Overview", "Product & Category", "Customer Analytics",
    "Advanced Analytics", "Operations & Quality"
])

# PAGE 1
with tabs[0]:
    st.subheader("Executive Sales Overview")
    cols = st.columns(4)
    with cols[0]: metric("Total Revenue", money(total_rev))
    with cols[1]: metric("Delivered Revenue", money(deliv_rev))
    with cols[2]: metric("Total Orders", f"{orders:,}")
    with cols[3]: metric("Delivered Orders", f"{deliv_orders:,}")
    cols = st.columns(4)
    with cols[0]: metric("Unique Customers", f"{customers:,}")
    with cols[1]: metric("Delivered AOV", money(aov))
    with cols[2]: metric("Quantity Sold", f"{filtered['quantity'].sum():,.0f}")
    with cols[3]:
        m = monthly_revenue(filtered)
        growth = np.nan
        if len(m) >= 2 and m.iloc[-2] != 0:
            growth = (m.iloc[-1] - m.iloc[-2]) / abs(m.iloc[-2]) * 100
        metric("Latest Month MoM", f"{growth:.1f}%" if pd.notna(growth) else "—")
    c1, c2 = st.columns(2)
    with c1:
        yr = filtered.dropna(subset=["order_date"]).groupby(filtered["order_date"].dt.year)["total_amount"].sum().reset_index()
        yr.columns = ["Year", "Revenue"]
        chart(px.line(yr, x="Year", y="Revenue", markers=True, title="Year-wise Revenue Trend"))
    with c2:
        mdf = m.rename("Revenue").reset_index()
        mdf.columns = ["Month", "Revenue"]
        chart(px.line(mdf, x="Month", y="Revenue", markers=True, title="Monthly Revenue Trend"))
    c1, c2 = st.columns(2)
    with c1:
        day = filtered.groupby("day_type_calc")["total_amount"].sum().reindex(["Weekday","Weekend"]).reset_index()
        day.columns = ["Day Type", "Revenue"]
        chart(px.bar(day, x="Day Type", y="Revenue", title="Weekday vs Weekend Revenue", color="Day Type"))
    with c2:
        if "season" in filtered:
            sea = filtered.groupby("season")["total_amount"].sum().reset_index()
            chart(px.bar(sea, x="season", y="total_amount", title="Festive vs Non-Festive Revenue"))
    c1, c2 = st.columns(2)
    with c1:
        status = filtered.groupby("order_status", dropna=False)["transaction_id"].nunique().reset_index(name="Orders")
        chart(px.pie(status, names="order_status", values="Orders", hole=.55, title="Order Status Distribution"))
    with c2:
        pay = filtered.groupby("payment", dropna=False)["total_amount"].sum().sort_values(ascending=False).head(10).reset_index()
        chart(px.bar(pay, x="total_amount", y="payment", orientation="h", title="Revenue by Payment Method"))


    # Dynamic written business insights for the currently selected filters
    year_rev = filtered.dropna(subset=["order_date"]).groupby(filtered["order_date"].dt.year)["total_amount"].sum().sort_index()
    mom_change, prev_month, latest_month = get_monthly_change(m)
    day_rev = filtered.groupby("day_type_calc")["total_amount"].sum()
    season_rev = filtered.groupby("season")["total_amount"].sum() if "season" in filtered else pd.Series(dtype=float)
    status_orders = filtered.groupby("order_status")["transaction_id"].nunique().sort_values(ascending=False)
    payment_rev = filtered.groupby("payment")["total_amount"].sum().sort_values(ascending=False)
    findings = [
        f"Selected data contains {orders:,} distinct transactions and {customers:,} customers, with recorded revenue of {money(total_rev)}.",
        f"Delivered transactions account for {money(deliv_rev)} across {deliv_orders:,} orders; delivered AOV is {money(aov)}.",
        (f"Revenue changed {signed_pct(mom_change)} from {prev_month:%b %Y} to {latest_month:%b %Y}." if prev_month is not None else "Month-over-month movement cannot be calculated reliably from fewer than two complete monthly observations or a zero comparison month.")
    ]
    if len(year_rev) >= 2:
        y0, y1 = year_rev.index[-2], year_rev.index[-1]
        ychg = safe_pct(year_rev.iloc[-1] - year_rev.iloc[-2], abs(year_rev.iloc[-2]))
        findings.append(f"Year-wise revenue changed {signed_pct(ychg)} from {int(y0)} to {int(y1)}.")
    if len(day_rev.dropna()) >= 2:
        best_day = day_rev.idxmax()
        findings.append(f"{best_day} revenue is higher in the selected period ({money(day_rev.max())}).")
    if len(season_rev):
        findings.append(f"Highest-revenue season is {season_rev.idxmax()} ({money(season_rev.max())}).")
    if len(status_orders):
        findings.append(f"Most common order status is {status_orders.index[0]} ({status_orders.iloc[0]:,} orders).")
    if len(payment_rev):
        findings.append(f"Top payment method by revenue is {payment_rev.index[0]} ({money(payment_rev.iloc[0])}).")
    interpretations = [
        "Delivered revenue and delivered AOV focus on completed orders, while total revenue includes all statuses. Read these measures together rather than treating them as interchangeable.",
        ("The latest monthly movement indicates a short-term change, not necessarily a sustained trend." if pd.notna(mom_change) else "The selected period does not provide enough comparable monthly data to infer short-term momentum."),
        "Differences by day, season, order status and payment method help identify where sales activity is concentrated; they do not by themselves explain the cause."
    ]
    actions = [
        "Review the months and segments behind any revenue decline before changing pricing or marketing.",
        "Investigate cancelled and other non-delivered orders to understand where order value is being lost.",
        "Plan campaigns and operational capacity around the strongest observed periods and monitor whether the pattern repeats."
    ]
    render_business_insights("Business Insights & Recommendations", findings, interpretations, actions)

# PAGE 2
with tabs[1]:
    st.subheader("Product & Category Performance")
    pcols = st.columns(4)
    with pcols[0]: metric("Products", f"{filtered['product_id'].nunique():,}")
    with pcols[1]: metric("Units Sold", f"{filtered['quantity'].sum():,.0f}")
    with pcols[2]: metric("Product Revenue", money(total_rev))
    with pcols[3]: metric("Average Unit Price", money(filtered["unit_price"].mean()))
    prod = filtered.groupby("product_name", dropna=False).agg(Revenue=("total_amount","sum"), Quantity=("quantity","sum")).reset_index()
    c1,c2 = st.columns(2)
    with c1:
        chart(px.bar(prod.nlargest(10,"Revenue").sort_values("Revenue"), x="Revenue", y="product_name", orientation="h", title="Top 10 Products by Revenue"))
    with c2:
        chart(px.bar(prod.nlargest(10,"Quantity").sort_values("Quantity"), x="Quantity", y="product_name", orientation="h", title="Top 10 Products by Quantity"))
    c1,c2 = st.columns(2)
    with c1:
        chart(px.bar(prod.nsmallest(10,"Revenue").sort_values("Revenue"), x="Revenue", y="product_name", orientation="h", title="Bottom 10 Products by Revenue"))
    with c2:
        cat = filtered.groupby("category", dropna=False)["total_amount"].sum().sort_values(ascending=False).reset_index()
        chart(px.pie(cat, names="category", values="total_amount", hole=.5, title="Category Revenue Contribution"))
    c1,c2 = st.columns(2)
    with c1:
        brand = filtered.groupby("brand", dropna=False)["total_amount"].sum().sort_values(ascending=False).head(12).reset_index()
        chart(px.bar(brand, x="total_amount", y="brand", orientation="h", title="Top Brands by Revenue"))
    with c2:
        disc = filtered.groupby("category", dropna=False).agg(Avg_Discount=("discount_pct","mean"), Revenue=("total_amount","sum")).reset_index()
        chart(px.scatter(disc, x="Avg_Discount", y="Revenue", size="Revenue", color="category", hover_name="category", title="Average Discount vs Revenue"))
    st.markdown("**Product performance detail**")
    st.dataframe(prod.sort_values("Revenue", ascending=False), use_container_width=True, height=300)


    # Dynamic product and category insights
    top_product = prod.sort_values("Revenue", ascending=False).head(1)
    bottom_product = prod.sort_values("Revenue", ascending=True).head(1)
    category_rev = filtered.groupby("category")["total_amount"].sum().sort_values(ascending=False)
    brand_rev = filtered.groupby("brand")["total_amount"].sum().sort_values(ascending=False)
    findings = [
        f"{filtered['product_id'].nunique():,} distinct products appear in the selected data, with {filtered['quantity'].sum():,.0f} units and {money(total_rev)} recorded revenue."
    ]
    if not top_product.empty:
        findings.append(f"Highest-revenue product is {top_product.iloc[0]['product_name']} ({money(top_product.iloc[0]['Revenue'])}).")
    if not bottom_product.empty:
        findings.append(f"Lowest-revenue product in the selected product list is {bottom_product.iloc[0]['product_name']} ({money(bottom_product.iloc[0]['Revenue'])}).")
    if len(category_rev):
        findings.append(f"Leading category is {category_rev.index[0]} ({money(category_rev.iloc[0])}, {safe_pct(category_rev.iloc[0], category_rev.sum()):.1f}% of category revenue).")
    if len(brand_rev):
        findings.append(f"Leading brand by revenue is {brand_rev.index[0]} ({money(brand_rev.iloc[0])}).")
    if "discount_pct" in filtered and filtered["discount_pct"].notna().any():
        findings.append(f"Average recorded discount is {filtered['discount_pct'].mean():.1f}%.")
    interpretations = [
        "Revenue leaders show where customer spending is concentrated; low-revenue products may reflect low demand, availability, price, or limited exposure.",
        "Category and brand concentration can create opportunities, but heavy dependence on a small set of products can also increase assortment risk.",
        "Discount and revenue moving together does not prove that discounts caused higher sales. Cost and margin data are not available, so profitability cannot be assessed."
    ]
    actions = [
        "Maintain availability of leading products and review stock levels against actual sales velocity.",
        "Investigate low-revenue products using stock availability, product age, pricing and customer feedback before discontinuing them.",
        "Test promotions by product/category and compare subsequent unit sales and revenue; avoid assuming a discount is profitable."
    ]
    render_business_insights("Business Insights & Recommendations", findings, interpretations, actions)

# PAGE 3
with tabs[2]:
    st.subheader("Customer & Retention Analytics")
    cust = filtered.groupby("customer_id", dropna=False).agg(
        Revenue=("total_amount","sum"), Orders=("transaction_id","nunique"),
        Last_Purchase=("order_date","max"), First_Purchase=("order_date","min")
    ).reset_index()
    repeat_n = int((cust["Orders"] > 1).sum())
    one_n = int((cust["Orders"] == 1).sum())
    ccols = st.columns(4)
    with ccols[0]: metric("Unique Customers", f"{len(cust):,}")
    with ccols[1]: metric("Repeat Customers", f"{repeat_n:,}")
    with ccols[2]: metric("One-time Customers", f"{one_n:,}")
    with ccols[3]: metric("Repeat Customer Rate", f"{repeat_n/len(cust)*100:.1f}%" if len(cust) else "—")
    c1,c2 = st.columns(2)
    with c1:
        topc = cust.nlargest(20,"Revenue").sort_values("Revenue")
        chart(px.bar(topc, x="Revenue", y="customer_id", orientation="h", title="Top 20 Customers by Revenue"))
    with c2:
        chart(px.pie(pd.DataFrame({"Type":["Repeat","One-time"],"Customers":[repeat_n,one_n]}), names="Type", values="Customers", hole=.55, title="Repeat vs One-time Customers"))
    c1,c2 = st.columns(2)
    with c1:
        seg = cust.copy()
        ref = filtered["order_date"].max()
        seg["Recency"] = (ref - seg["Last_Purchase"]).dt.days
        seg["Segment"] = np.select(
            [(seg["Recency"] >= 180) & (seg["Orders"] >= 2),
             (seg["Recency"] >= 90) & (seg["Orders"] >= 2),
             (seg["Recency"] < 90) & (seg["Orders"] >= 4),
             (seg["Recency"] >= 180) & (seg["Orders"] == 1)],
            ["At-risk Customers","Loyal Customers","Champions","Lost Customers"], default="Others"
        )
        sc = seg["Segment"].value_counts().reset_index()
        sc.columns = ["Segment","Customers"]
        chart(px.bar(sc, x="Segment", y="Customers", title="Customer Segmentation (dashboard heuristic)"))
    with c2:
        age = filtered.groupby("age_group", dropna=False)["total_amount"].sum().reset_index()
        chart(px.bar(age, x="age_group", y="total_amount", title="Revenue by Age Group"))
    c1,c2 = st.columns(2)
    with c1:
        gender = filtered.groupby("gender", dropna=False)["total_amount"].sum().reset_index()
        chart(px.bar(gender, x="gender", y="total_amount", title="Revenue by Gender"))
    with c2:
        freq = cust["Orders"].value_counts().sort_index().reset_index()
        freq.columns = ["Orders per Customer","Customers"]
        chart(px.bar(freq, x="Orders per Customer", y="Customers", title="Purchase Frequency Distribution"))
    st.caption("RFM-style segments shown here are a transparent dashboard heuristic; use your validated RFM scoring table if you have exported it separately.")


    # Dynamic customer insights
    repeat_rate = safe_pct(repeat_n, len(cust))
    cust_revenue_total = cust["Revenue"].sum()
    top_customer_share = safe_pct(cust.nlargest(min(20, len(cust)), "Revenue")["Revenue"].sum(), cust_revenue_total) if len(cust) else np.nan
    findings = [
        f"The filtered dataset contains {len(cust):,} customers: {repeat_n:,} repeat customers and {one_n:,} one-time customers.",
        f"Repeat-customer rate is {repeat_rate:.1f}%." if pd.notna(repeat_rate) else "Repeat-customer rate is not available.",
        f"The top 20 customers account for {top_customer_share:.1f}% of customer revenue." if pd.notna(top_customer_share) else "Top-customer revenue share is not available."
    ]
    if len(freq):
        most_common_frequency = freq.sort_values("Customers", ascending=False).iloc[0]
        findings.append(f"Most common purchase frequency is {int(most_common_frequency['Orders per Customer'])} order(s) per customer.")
    if len(age):
        age_top = age.sort_values("total_amount", ascending=False).iloc[0]
        findings.append(f"Highest-revenue age group is {age_top['age_group']} ({money(age_top['total_amount'])}).")
    if len(gender):
        gender_top = gender.sort_values("total_amount", ascending=False).iloc[0]
        findings.append(f"Highest-revenue gender group is {gender_top['gender']} ({money(gender_top['total_amount'])}).")
    interpretations = [
        "A higher repeat-customer rate indicates more customers have purchased more than once within the selected data window; it is not a lifetime retention measure.",
        "Revenue concentration among top customers can create value but also means a small group may contribute a large share of sales.",
        "Age and gender revenue differences describe this dataset and should not be treated as proof of customer preferences or causes."
    ]
    actions = [
        "Create a follow-up campaign for one-time customers and measure how many return for another purchase.",
        "Use customer purchase history to test relevant loyalty benefits and personalized communication.",
        "Monitor top-customer concentration and avoid relying on a small number of customers for the entire sales target."
    ]
    render_business_insights("Business Insights & Recommendations", findings, interpretations, actions)

# PAGE 4
with tabs[3]:
    st.subheader("Advanced Analytics & Forecasting")
    ref = filtered["order_date"].max()
    cust_risk = filtered.groupby("customer_id").agg(Last_Purchase=("order_date","max"), Revenue=("total_amount","sum"), Orders=("transaction_id","nunique"))
    cust_risk["Inactive_Days"] = (ref - cust_risk["Last_Purchase"]).dt.days
    cust_risk["Risk"] = np.select([cust_risk["Inactive_Days"] >= 180, cust_risk["Inactive_Days"] >= 90], ["High Risk","Medium Risk"], default="Active")
    risk_counts = cust_risk["Risk"].value_counts().reindex(["Active","Medium Risk","High Risk"], fill_value=0).reset_index()
    risk_counts.columns = ["Risk","Customers"]
    # IQR flags on non-null numeric values
    q1q, q3q = filtered["quantity"].quantile(.25), filtered["quantity"].quantile(.75)
    q1a, q3a = filtered["total_amount"].quantile(.25), filtered["total_amount"].quantile(.75)
    uq, ua = q3q + 1.5*(q3q-q1q), q3a + 1.5*(q3a-q1a)
    anomalies = filtered[(filtered["quantity"] > uq) | (filtered["total_amount"] > ua)].copy()
    acols = st.columns(4)
    with acols[0]: metric("Active Customers", f"{int((cust_risk['Risk']=='Active').sum()):,}")
    with acols[1]: metric("Medium Risk", f"{int((cust_risk['Risk']=='Medium Risk').sum()):,}")
    with acols[2]: metric("High Risk", f"{int((cust_risk['Risk']=='High Risk').sum()):,}")
    with acols[3]: metric("IQR Flagged Transactions", f"{len(anomalies):,}")
    c1,c2 = st.columns(2)
    with c1:
        chart(px.pie(risk_counts, names="Risk", values="Customers", hole=.55, title="Customer Inactivity Risk"))
    with c2:
        chart(px.bar(risk_counts, x="Risk", y="Customers", title="Customer Risk Counts"))
    m = monthly_revenue(filtered).dropna()
    if len(m) >= 6:
        X = np.arange(len(m)).reshape(-1,1)
        y = m.values
        model = LinearRegression().fit(X,y)
        future_x = np.arange(len(m),len(m)+3).reshape(-1,1)
        pred = model.predict(future_x)
        future_dates = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        fdf = pd.DataFrame({"Month":future_dates,"Revenue":pred,"Type":"Forecast"})
        adf = m.rename("Revenue").reset_index()
        adf.columns = ["Month","Revenue"]
        adf["Type"] = "Actual"
        plotdf = pd.concat([adf,fdf],ignore_index=True)
        chart(px.line(plotdf, x="Month", y="Revenue", color="Type", markers=True, title="Actual Monthly Revenue & 3-Month Linear Trend Forecast"))
        st.caption("Forecast is a simple linear-regression baseline, not a validated prediction interval. Compare it with a chronological holdout before using it as a business commitment.")
    else:
        st.info("At least six monthly observations are required for this baseline forecast.")
    cust_rev = filtered.groupby("customer_id")["total_amount"].sum().sort_values(ascending=False)
    cum = pd.Series(dtype=float)
    if len(cust_rev):
        cum = cust_rev.cumsum()/cust_rev.sum()*100
        pareto = pd.DataFrame({"Customer Rank":np.arange(1,len(cum)+1),"Cumulative Revenue %":cum.values})
        pareto = pareto.iloc[::max(1,len(pareto)//500)].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pareto["Customer Rank"], y=pareto["Cumulative Revenue %"], mode="lines", name="Cumulative Revenue"))
        fig.add_hline(y=80, line_dash="dash", line_color=RED)
        fig.update_layout(title="Customer Revenue Pareto Curve", xaxis_title="Customer Rank", yaxis_title="Cumulative Revenue %")
        chart(fig)
    st.markdown("**IQR-flagged transaction sample**")
    cols_show = [c for c in ["transaction_id","customer_id","product_name","quantity","unit_price","total_amount"] if c in anomalies]
    st.dataframe(anomalies[cols_show].head(500), use_container_width=True, height=280)
    st.caption(f"IQR thresholds in current filter context: quantity > {uq:.2f}; transaction amount > {ua:.2f}. These are statistical flags, not proof of fraud.")


    # Dynamic advanced analytics insights
    high_risk_n = int((cust_risk["Risk"] == "High Risk").sum())
    medium_risk_n = int((cust_risk["Risk"] == "Medium Risk").sum())
    active_n = int((cust_risk["Risk"] == "Active").sum())
    risk_base = len(cust_risk)
    risk_pct = safe_pct(high_risk_n + medium_risk_n, risk_base)
    flagged_pct = safe_pct(len(anomalies), len(filtered))
    pareto_80_customers = int((cum < 80).sum() + 1) if len(cum) and cum.iloc[-1] >= 80 else len(cum) if len(cum) else 0
    findings = [
        f"Current inactivity heuristic classifies {high_risk_n:,} customers as high risk, {medium_risk_n:,} as medium risk and {active_n:,} as active.",
        f"High/medium inactivity-risk customers represent {risk_pct:.1f}% of customers." if pd.notna(risk_pct) else "Customer inactivity-risk percentage is unavailable.",
        f"IQR rules flag {len(anomalies):,} transactions ({flagged_pct:.1f}% of filtered transactions)." if pd.notna(flagged_pct) else "IQR flag rate is unavailable."
    ]
    if len(m) >= 6:
        findings.append(f"The three-month linear-trend baseline forecast is {money(np.sum(pred))} in total; it is an indicative baseline, not a guaranteed outcome.")
    else:
        findings.append("A three-month forecast is not shown because at least six monthly observations are required.")
    if len(cum):
        findings.append(f"Approximately {pareto_80_customers:,} top customers reach 80% of cumulative customer revenue in this filtered dataset.")
    interpretations = [
        "Inactivity risk is based on recency thresholds (90/180 days) and is a screening heuristic, not a trained churn prediction.",
        "IQR flags identify unusually high transaction quantities or values for review; they are not evidence of fraud or errors.",
        "The linear forecast extrapolates a simple time trend and does not account for seasonality, promotions or external factors."
    ]
    actions = [
        "Prioritize a small, measurable re-engagement test for high-risk customers and track their return-purchase rate.",
        "Review flagged transactions against source records and business context before taking action.",
        "Compare forecast results with actual future revenue and a chronological holdout before using them for planning."
    ]
    render_business_insights("Business Insights & Recommendations", findings, interpretations, actions)

# PAGE 5
with tabs[4]:
    st.subheader("Operations, Geography & Data Quality")
    avg_delivery = filtered["delivery_days"].where(filtered["delivery_days"] >= 0).mean()
    avg_rating = filtered["rating"].mean()
    ocols = st.columns(4)
    with ocols[0]: metric("Average Delivery Days", f"{avg_delivery:.1f}" if pd.notna(avg_delivery) else "—")
    with ocols[1]: metric("Average Rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "—")
    with ocols[2]: metric("Missing Order Dates", f"{filtered['order_date'].isna().sum():,}")
    with ocols[3]: metric("Duplicate Transaction IDs", f"{filtered['transaction_id'].duplicated().sum():,}")
    c1,c2 = st.columns(2)
    with c1:
        state_rev = filtered.groupby("state", dropna=False)["total_amount"].sum().sort_values(ascending=False).head(15).reset_index()
        chart(px.bar(state_rev, x="total_amount", y="state", orientation="h", title="Top States by Revenue"))
    with c2:
        delivery = filtered[filtered["delivery_days"].between(0,90)]
        chart(px.histogram(delivery, x="delivery_days", nbins=25, title="Delivery Time Distribution (0–90 days)"))
    c1,c2 = st.columns(2)
    with c1:
        rating = filtered["rating"].value_counts(dropna=False).sort_index().reset_index()
        rating.columns = ["Rating","Count"]
        chart(px.bar(rating, x="Rating", y="Count", title="Customer Rating Distribution"))
    with c2:
        status_month = filtered.dropna(subset=["order_date"]).groupby([filtered["order_date"].dt.to_period("M").astype(str),"order_status"])["transaction_id"].nunique().reset_index()
        status_month.columns = ["Month","Order Status","Orders"]
        chart(px.bar(status_month, x="Month", y="Orders", color="Order Status", title="Monthly Order Status Trend"))
    missing = filtered.isna().mean().mul(100).sort_values(ascending=False).head(15).reset_index()
    missing.columns = ["Column","Missing %"]
    chart(px.bar(missing.sort_values("Missing %"), x="Missing %", y="Column", orientation="h", title="Top Missing-Value Columns"))
    st.markdown("**Filtered transaction data**")
    st.dataframe(filtered.head(1000), use_container_width=True, height=300)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_retail_data.csv", "text/csv")

    # Dynamic operations and data-quality insights
    findings = [
        f"Average observed delivery time is {avg_delivery:.1f} days." if pd.notna(avg_delivery) else "Average delivery time is unavailable for the selected records.",
        f"Average customer rating is {avg_rating:.2f} out of 5." if pd.notna(avg_rating) else "Average customer rating is unavailable.",
        f"{filtered['order_date'].isna().sum():,} rows have missing order dates and {filtered['transaction_id'].duplicated().sum():,} transaction IDs are repeated in the filtered rows."
    ]
    if len(state_rev):
        findings.append(f"Highest-revenue state is {state_rev.iloc[0]['state']} ({money(state_rev.iloc[0]['total_amount'])}).")
    if len(status_month):
        last_status_month = status_month["Month"].max()
        findings.append(f"Monthly order-status chart includes data through {last_status_month}.")
    interpretations = [
        "Delivery duration helps describe fulfilment experience, but no promised delivery date is supplied, so on-time delivery performance cannot be measured.",
        "Ratings provide a customer-feedback signal; investigate low-rating patterns by product, location and order status where sample sizes are sufficient.",
        "Repeated transaction IDs and missing dates should be reviewed in context; repeated IDs may represent multiple line items rather than duplicate orders."
    ]
    actions = [
        "Review unusually long delivery durations by state or city and investigate the underlying fulfilment stages.",
        "Inspect low-rating products and customer comments if available, then monitor rating changes after improvements.",
        "Validate repeated transaction IDs against the data grain before removing anything, and investigate missing dates at source."
    ]
    render_business_insights("Business Insights & Recommendations", findings, interpretations, actions)



st.divider()
st.caption("Metric note: Total transaction value uses the supplied total_amount field. Delivered revenue and delivered AOV are calculated using Delivered records only. Profit is not calculated because product cost is not supplied.")
