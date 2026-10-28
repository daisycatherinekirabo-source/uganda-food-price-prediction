import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Uganda Food Price Predictor", page_icon="🇺🇬", layout="wide")

DATA_FILE = "wfp_food_prices.csv"   # same file you used in Colab, saved next to app.py
TEST_FROM = 2023                    # years from here on are held out to measure accuracy
BLACKLIST = ['fuel', 'petrol', 'diesel', 'wage', 'bins', 'underwear', 'sticks',
             'soap', 'charcoal', 'firewood', 'jerrycan', 'plastic', 'kerosene']
CAT_COLS = {'admin1': 'admin1_n', 'market': 'market_n', 'commodity': 'commodity_n', 'unit': 'unit_n'}
FEATS = ['month', 'admin1_n', 'market_n', 'commodity_n', 'unit_n']


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    raw = pd.read_csv(DATA_FILE)
    df = raw[~raw['commodity'].str.contains('|'.join(BLACKLIST), case=False, na=False)].copy()
    df = df.drop_duplicates()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['t'] = df['year'] + (df['month'] - 1) / 12          # time as one number
    df['log_price'] = np.log1p(df['price'])                # steady % growth on a log scale
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# Model: straight-line price trend per food item + Random Forest for
# market, food and month effects (so future years can keep trending)
# ---------------------------------------------------------------
def fit_trends(train, min_rows=30, min_years=4):
    g = LinearRegression().fit(train[['t']], train['log_price'])
    global_trend = (g.coef_[0], g.intercept_)
    trends = {}
    for key, grp in train.groupby(['commodity_n', 'unit_n']):
        if len(grp) >= min_rows and grp['year'].nunique() >= min_years:
            m = LinearRegression().fit(grp[['t']], grp['log_price'])
            trends[key] = (m.coef_[0], m.intercept_)
    return trends, global_trend


def trend_value(frame, trends, global_trend):
    out = np.empty(len(frame))
    for i, (c, u, t) in enumerate(zip(frame['commodity_n'], frame['unit_n'], frame['t'])):
        slope, intercept = trends.get((c, u), global_trend)
        out[i] = slope * t + intercept
    return out


def fit_hybrid(train):
    trends, global_trend = fit_trends(train)
    residual = train['log_price'] - trend_value(train, trends, global_trend)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(train[FEATS], residual)
    return trends, global_trend, rf


def hybrid_predict(frame, model):
    trends, global_trend, rf = model
    return np.expm1(trend_value(frame, trends, global_trend) + rf.predict(frame[FEATS]))


@st.cache_resource(show_spinner="Training the model (this takes a short while the first time)...")
def build_model():
    df = load_data()
    encoders = {}
    for col, coded in CAT_COLS.items():
        encoders[col] = LabelEncoder().fit(df[col])
        df[coded] = encoders[col].transform(df[col])

    score = None                                           # accuracy on years the model has not seen
    train_df, test_df = df[df['year'] < TEST_FROM], df[df['year'] >= TEST_FROM]
    if len(train_df) > 100 and len(test_df) > 100:
        pred = hybrid_predict(test_df, fit_hybrid(train_df))
        actual = test_df['price']
        score = (r2_score(actual, pred) * 100, np.mean(np.abs((actual - pred) / actual)) * 100)

    return df, encoders, fit_hybrid(df), score


# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
df, encoders, model, score = build_model()
last_date = df['date'].max()

st.title("🇺🇬 Uganda Food Price Predictor")
st.caption("Automated price estimates for market resilience and food security. "
           f"Based on WFP market data, {df['year'].min()} to {last_date.year}.")

with st.sidebar:
    st.subheader("About the model")
    st.write("Price trend per food item + Random Forest for market, food and month effects.")
    if score:
        st.write(f"Tested on unseen years ({TEST_FROM}+): R² {score[0]:.1f}%, average error {score[1]:.1f}%.")
    st.write("Estimates beyond the data follow the historical price trend. They are estimates, not guarantees.")

left, right = st.columns([1, 1])

with left:
    region = st.selectbox("1. Region", sorted(df['admin1'].unique()))
    reg_df = df[df['admin1'] == region]
    market = st.selectbox("2. Market", sorted(reg_df['market'].unique()))
    mkt_df = reg_df[reg_df['market'] == market]
    food = st.selectbox("3. Food item", sorted(mkt_df['commodity'].unique()))
    food_df = mkt_df[mkt_df['commodity'] == food]
    unit = st.selectbox("4. Unit of measurement", sorted(food_df['unit'].unique()))
    sel = food_df[food_df['unit'] == unit]

    c1, c2 = st.columns(2)
    year = c1.number_input("Year", min_value=int(df['year'].min()), max_value=2040, value=2030, step=1)
    month = c2.slider("Month", 1, 12, 8)

codes = {coded: int(encoders[col].transform([{'admin1': region, 'market': market,
                                              'commodity': food, 'unit': unit}[col]])[0])
         for col, coded in CAT_COLS.items()}

# One estimate for the chosen date, plus a month-by-month path for the chart
target = pd.Period(year=int(year), month=int(month), freq='M')
start = sel['date'].min().to_period('M')
end = max(target, sel['date'].max().to_period('M'))
periods = pd.period_range(start, end, freq='M')
grid = pd.DataFrame({'year': periods.year, 'month': periods.month})
grid['t'] = grid['year'] + (grid['month'] - 1) / 12
for coded, value in codes.items():
    grid[coded] = value
grid['Model estimate'] = hybrid_predict(grid, model)

one = pd.DataFrame({'year': [target.year], 'month': [target.month],
                    't': [target.year + (target.month - 1) / 12], **{k: [v] for k, v in codes.items()}})
price = float(hybrid_predict(one, model)[0])

with right:
    st.metric(f"Estimated price for {target.strftime('%B %Y')}", f"{price:,.0f} UGX",
              help=f"{food}, per {unit}, {market}")
    latest = sel.sort_values('date').iloc[-1]
    st.caption(f"Latest recorded price: {latest['price']:,.0f} UGX ({latest['date']:%B %Y}).")
    if target.to_timestamp() > last_date:
        st.info(f"This date is beyond the last data point ({last_date:%B %Y}), "
                "so the estimate follows the historical price trend.")

    history = sel.groupby(sel['date'].dt.to_period('M'))['price'].mean().reindex(periods).to_numpy()
    chart = pd.DataFrame({'Actual price (WFP data)': history,
                          'Model estimate': grid['Model estimate'].to_numpy()},
                         index=periods.to_timestamp())
    st.line_chart(chart)
