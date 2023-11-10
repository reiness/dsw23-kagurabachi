import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import numpy as np
import folium
import h3
from branca.colormap import LinearColormap
import io
import seaborn as sns
import plotly.graph_objects as go
import matplotlib as plt
import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

#Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Set page configuration
st.set_page_config(
    page_title="Data Science Week",
    page_icon="✅",
    layout="wide",
)

# Define the EDA tab
def eda_tab():
    st.header("Exploratory Data Analysis")
    st.write('---')
    data = pd.read_excel('data.xlsx')

    st.subheader("DataFrame Info")
    nrows = st.number_input("Show number of rows", min_value=1, value=5, step=1)
    st.dataframe(data.head(nrows), use_container_width=True)
    
    # Capture the .info() output into a string
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_text = buffer.getvalue()
    buffer.close()
    
    # Display .info() of the DataFrame as Markdown
    with st.expander("Click here to see the detailed information"):
        st.markdown(f"```{info_text}```", unsafe_allow_html=True)

    st.subheader("General Churn Rate")
    fig = px.pie(data.groupby('Churn Label')['Customer ID'].nunique().reset_index(), 
         values='Customer ID', 
         names='Churn Label')
    st.plotly_chart(fig)

    st.subheader("Map of Churn Rates by Hexagon")
    hex_level = 5
    data['hex_id'] = data.apply(lambda x: h3.geo_to_h3(x['Latitude'], x['Longitude'], hex_level), axis=1)
    churn = data.assign(churn_clients = np.where(data['Churn Label']=='Yes',data['Customer ID'],None)).groupby(['hex_id']).agg({'churn_clients':'count'}).reset_index()
    clients = data.groupby(['hex_id'])['Customer ID'].count().reset_index()
    churn_data = clients.join(churn.set_index(['hex_id']), on=['hex_id'])
    churn_data['churn_rate'] = churn_data['churn_clients']/churn_data['Customer ID']
    churn_data['center'] = churn_data['hex_id'].apply(lambda x: h3.h3_to_geo(x))

    color_range = [churn_data['churn_rate'].min(), churn_data['churn_rate'].max()]
    colormap = LinearColormap(["green", "orange", "red"], vmin=min(color_range), vmax=max(color_range))

    mean_lat, mean_lon = churn_data['center'].apply(lambda x: x[0]).mean(), churn_data['center'].apply(lambda x: x[1]).mean()
    map_center = [mean_lat, mean_lon]

    # Create a map
    m = folium.Map(location=map_center, zoom_start=9, width='100%', height='80%', tiles='OpenStreetMap')

    for _, row in churn_data.iterrows():
        folium.Polygon(
            locations=h3.h3_to_geo_boundary(row['hex_id']),
            fill=True,
            fill_color=colormap(row['churn_rate']),
            fill_opacity=0.7,
            stroke=False,
            tooltip=f"Churn rate: {row['churn_rate']}<br>Number of customers: {row['Customer ID']}"
        ).add_to(m)

    colormap.caption = 'Churn rate'
    m.add_child(colormap)

    
    st.write("This map visualizes churn rates by hexagon.")
    st.write("Green represents lower churn rates, while red indicates higher churn rates.")
    components.html(m._repr_html_(), height=700)


    st.subheader("Customer's lifetime in the service")
    fig = px.histogram(data, x="Tenure Months", color="Churn Label",marginal="box" )
    fig.update_layout(
        width=1100,  # Adjust the width
        height=900,  # Adjust the height
    )
    st.plotly_chart(fig)

    with st.expander("Click here to see the 'Churn Label' statistics"):
        st.subheader("Quantiles of Tenure Months")
        quantiles = data.groupby('Churn Label')['Tenure Months'].quantile([0.50, 0.75, 0.90, 0.95])
        st.write(quantiles)

        st.subheader("Mean of Tenure Months")
        means = data.groupby('Churn Label')['Tenure Months'].mean()
        st.write(means)


    st.subheader("Services used by Client")
    corr_df = data.copy()
    corr_df['Churn Label'].replace(to_replace='Yes', value=1, inplace=True)
    corr_df['Churn Label'].replace(to_replace='No',  value=0, inplace=True)
    column_subset = corr_df[['Churn Label','Device Class', 'Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp', 'Payment Method']]
    df_dummies = pd.get_dummies(column_subset, dtype=int)
    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df_dummies.corr(),
        x=df_dummies.columns,
        y=df_dummies.columns,
        colorscale='picnic',
    ))

    # Adjust the size of the heatmap within the layout
    fig.update_layout(
        width=1100,  # Adjust the width
        height=900,  # Adjust the height
    )
    # Display the Plotly heatmap using st.plotly_chart
    st.plotly_chart(fig)

    fig = px.bar(df_dummies.corr()['Churn Label'].sort_values(ascending = False), 
             color = 'value')
    
    fig.update_layout(
        width=1100,  # Adjust the width
        height=900,  # Adjust the height
    )
    st.plotly_chart(fig)

    fig = px.pie(data.groupby(['Device Class','Churn Label'])['Customer ID'].count().reset_index(), 
             values='Customer ID', 
             facet_col = 'Churn Label',
             names='Device Class',
            title = "What type of clients' devices who left the service?")
    fig.update_layout(
        width=1000,  # Adjust the width
        height=800,  # Adjust the height
    )
    st.plotly_chart(fig)

    fig = px.pie(data.groupby(['Payment Method','Churn Label'])['Customer ID'].count().reset_index(), 
             values='Customer ID', 
             facet_col = 'Churn Label',
             names='Payment Method',
             hole = .5,
            title = "What type of clients' payment method who left the service?")
    fig.update_layout(
        width=1000,  # Adjust the width
        height=800,  # Adjust the height
    )
    st.plotly_chart(fig)

    fig = px.pie(data.groupby(['Payment Method','Churn Label'])['Customer ID'].count().reset_index(), 
            values='Customer ID', 
            names='Churn Label',
            facet_col = 'Payment Method',
            color = 'Churn Label',
            title = 'Churn rate by customer payment method')
    fig.update_layout(
        width=1000,  # Adjust the width
        height=800,  # Adjust the height
    )
    st.plotly_chart(fig)

def chat_tab():
    plt.use('TkAgg')

    API_KEY = st.secrets['OPENAI_API_KEY']
    llm = OpenAI(api_token=API_KEY)

    df = pd.read_excel('data.xlsx')
    df = SmartDataframe(df, config={'llm':llm})

    st.header("Hello, I'm Kagura-Bot!")
    st.image('maid.png',width=400)
    

    with st.form("prompt_area"):
        prompt = st.text_input("Ask me anything ^ ^")
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        if prompt:
            with st.spinner("Generating answer, please wait..."):
                st.write(df.chat(prompt))
        else:
            st.write("Please enter a request.")

def statistic_test_tab():
    st.header('Statistic Tests')
    st.write('---')

def model_tab():
    st.header('Model')
    st.write('---')

    # Load data
    df = pd.read_excel('data.xlsx')
    
    # Preprocess data
    df.drop(columns=['Customer ID', 'Longitude', 'Latitude'], inplace=True)
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Churn Label']), df['Churn Label'], test_size=0.2, random_state=42, stratify=df['Churn Label'])

    # Apply Power Transformer
    numeric = ['Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']
    scaler_power = PowerTransformer(method='yeo-johnson')
    X_train[numeric] = scaler_power.fit_transform(X_train[numeric])
    X_test[numeric] = scaler_power.fit_transform(X_test[numeric])

    # Apply SMOTE
    smote = SMOTEN(random_state=42, k_neighbors=3)
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    # Train RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_estimators=1460,
        min_samples_split=29,
        min_samples_leaf=18,
        max_depth=97,
        criterion='gini'
    )
    rf_classifier.fit(X_train_over, y_train_over)

    st.write('---')
    st.header('Input Predict')
    
    # User Input for Prediction
    tenure_months = st.number_input('Tenure Months:')
    location = st.selectbox('Location:', df['Location'].unique())
    device_class = st.selectbox('Device Class:', df['Device Class'].unique())
    games_product = st.selectbox('Games Product:', ['Yes', 'No'])
    music_product = st.selectbox('Music Product:', ['Yes', 'No'])
    education_product = st.selectbox('Education Product:', ['Yes', 'No'])
    call_center = st.selectbox('Call Center:', ['Yes', 'No'])
    video_product = st.selectbox('Video Product:', ['Yes', 'No'])
    use_myapp = st.selectbox('Use MyApp:', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method:', df['Payment Method'].unique())
    monthly_purchase = st.number_input('Monthly Purchase (Thou. IDR):')
    cltv = st.number_input('CLTV (Predicted Thou. IDR):')

    # Preprocess user input
    input_data = pd.DataFrame({
        'Tenure Months': [tenure_months],
        'Location': [location],
        'Device Class': [device_class],
        'Games Product': [games_product],
        'Music Product': [music_product],
        'Education Product': [education_product],
        'Call Center': [call_center],
        'Video Product': [video_product],
        'Use MyApp': [use_myapp],
        'Payment Method': [payment_method],
        'Monthly Purchase (Thou. IDR)': [monthly_purchase],
        'CLTV (Predicted Thou. IDR)': [cltv]
    })

    # Transformasi label encoding pada input_data
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            input_data[col] = le.transform(input_data[col])

    input_data[numeric] = scaler_power.transform(input_data[numeric])

    # Make Prediction
    prediction = rf_classifier.predict(input_data)

    st.write('---')
    st.header('Prediction Result')

    # Mengubah nilai 0 menjadi 'No' dan nilai 1 menjadi 'Yes'
    prediction_label = 'Yes' if prediction[0] == 1 else 'No'

    st.write(f'The predicted Churn Label is: {prediction_label}')

    st.write('---')
    st.header('Classification Report')
    y_pred = rf_classifier.predict(X_test)
    classification_rep = classification_report(y_test, y_pred)
    st.text(classification_rep)


# Create the sidebar for content selection
selected_tab = st.sidebar.selectbox("Select a tab:", ["EDA", "Chat", "Statistic Test", "Model"])

# Display content based on the selected tab
if selected_tab == "EDA":
    eda_tab()
elif selected_tab == "Chat":
    chat_tab()
elif selected_tab == "Statistic Test":
    statistic_test_tab()
else:
    model_tab()
