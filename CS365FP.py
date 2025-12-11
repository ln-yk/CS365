import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# PAGE CONFIG & CUSTOM CSS THEME
st.set_page_config(page_title="Startup Profit Predictor", layout="wide")

st.markdown(
    """
    <style>
        /* GLOBAL THEME */
        body {
            background-color: #F7F7F9;
        }

        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #282C49;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* TITLES */
        h1, h2, h3 {
            color: white !important;
        }

        /* BUTTONS */
        div.stButton > button {
            background-color: #A47488;
            color: white;
            border-radius: 8px;
            height: 45px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #E9B1A4;
            color: black;
        }

        /* TABLES */
        .stDataFrame, .stTable {
            border: 1px solid #A47488;
            border-radius: 8px;
        }

        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #E9B1A4 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #A47488;
            color: white !important;
        }

        /* HR LINE */
        hr {
            border: 1px solid #A47488;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# LOAD DATA
try:
    df = pd.read_csv('50_Startups.csv')
except FileNotFoundError:
    st.error("Error: Please place '50_Startups.csv' in the same directory as this script.")
    st.stop()

df_processed = pd.get_dummies(df, columns=['State'], drop_first=True)
X = df_processed.drop(columns=['Profit'])
y = df_processed['Profit']

model = LinearRegression()
model.fit(X, y)

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Data Exploration & Preparation",
        "Analysis & Insights",
        "Conclusions & Recommendations"
    ]
)

# OVERVIEW SECTION
if page == "Overview":
    st.title("Overview")
    st.write("""
    This project analyzes the **50 Startups Dataset** to predict the annual **Profit** 
    based on operational spending and location.
    
    We use **Multiple Linear Regression** to:
    - Identify influential spending categories  
    - Evaluate the effect of state location  
    - Predict profits through user-provided inputs  
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe(include="all"))

# EXPLORATION SECTION
elif page == "Data Exploration & Preparation":
    st.title("Data Exploration & Preparation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)

    st.subheader("Feature Distribution")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].hist(bins=10, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ANALYSIS & INSIGHTS SECTION
elif page == "Analysis & Insights":
    st.title("Analysis & Insights")
    st.write("Customize inputs below to generate a profit prediction.")

    # inputs
    st.subheader("Profit Prediction Inputs")

    colA, colB, colC, colD = st.columns(4)

    with colA:
        rd_spend = st.number_input("R&D Spend ($)", value=100000.0, step=5000.0)

    with colB:
        admin_spend = st.number_input("Admin Spend ($)", value=50000.0, step=5000.0)

    with colC:
        marketing_spend = st.number_input("Marketing Spend ($)", value=200000.0, step=5000.0)

    with colD:
        selected_state = st.selectbox("State", df['State'].unique())

    predict_button = st.button("Predict Profit")

    if predict_button:
        input_data = pd.DataFrame({
            "R&D Spend": [rd_spend],
            "Administration": [admin_spend],
            "Marketing Spend": [marketing_spend],
            "State": [selected_state]
        })

        # process input
        input_processed = pd.get_dummies(input_data, columns=['State'], drop_first=True)
        input_processed = input_processed.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_processed)[0]

        st.success(f"Predicted Annual Profit: **${prediction:,.2f}**")

        if prediction > 150000:
            st.balloons()
            st.info("High Profit Potential Detected.")
        else:
            st.warning("Moderate prediction. Consider increasing R&D investment.")

    # VISUALIZATION TABS
    st.markdown("---")
    st.subheader("Visual Insights")

    tab1, tab2, tab3 = st.tabs(["R&D vs Profit", "Marketing vs Profit", "Regression Coefficients"])

    with tab1:
        st.scatter_chart(df, x="R&D Spend", y="Profit")

    with tab2:
        st.scatter_chart(df, x="Marketing Spend", y="Profit")

    with tab3:
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        })
        st.table(coef_df)

    # MAP VISUALIZATION
    st.markdown("---")
    st.subheader("Geographical Representation of States")

    centers = {
        "Florida": (28.0, -82.0),
        "New York": (42.9, -75.0),
        "California": (37.3, -119.7)
    }

    def generate_cluster(center_lat, center_lon, n=350, spread=6):
        pts = np.random.randn(n, 2) / spread + [center_lat, center_lon]
        return pd.DataFrame(pts, columns=["lat", "lon"])

    fl = generate_cluster(*centers["Florida"])
    ny = generate_cluster(*centers["New York"])
    ca = generate_cluster(*centers["California"])

    map_df = pd.concat([fl, ny, ca], ignore_index=True)
    st.map(map_df, zoom=3)

# CONCLUSIONS SECTION
elif page == "Conclusions & Recommendations":
    st.title("Conclusions & Recommendations")

    st.write("""
    ### Key Conclusions
    - R&D Spending drives the most profit impact.
    - Marketing has moderate influence; Administration contributes the least.
    - State differences have subtle but noticeable effects.
    - Linear Regression provides reliable financial estimation.

    ### Recommendations
    - Prioritize **R&D investment**.
    - Monitor ROI before increasing marketing budgets.
    - Limit unnecessary administrative costs.
    - Consider profitability differences when choosing a state.
    """)

