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
from sklearn.ensemble import IsolationForest
import numpy as np
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.stats import shapiro
from scipy.stats import f_oneway
from scipy.stats import spearmanr
import subprocess
subprocess.run(["pip", "install", "--upgrade", "pandasai"])

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
    # plt.use('TkAgg')

    API_KEY = st.secrets['OPENAI_API_KEY']
    llm = OpenAI(api_token=API_KEY)

    df = pd.read_excel('data.xlsx')
    df = SmartDataframe(df, config={'llm':llm})

    st.header("Hello, I'm Kagura-chan!")
    st.image('kagurachan.png', width=400)

    keyword = ['viz', 'visualization', 'plot', 'barplot', 'graf', 'graph', 'visualisasi','chart','gambar']

    with st.form("prompt_area"):
        prompt = st.text_input("Ask me anything ^ ^")
        submitted = st.form_submit_button("Ask Kagura-chan")
        # st.image(st.secrets['KAGURA_VIZ'], width=500, caption="Kagura-chan's artwork") 

    if submitted:
            if prompt:
                with st.spinner("Kagura-chan is thinking (⌐■_■) , please wait..."):
                    bot_out = df.chat(prompt)
                    if bot_out is not None:
                        st.write(bot_out)
                    else:
                        st.write('\n')

                    # st.write(df.chat(prompt))
                    if any(kw in prompt.lower() for kw in keyword):
                        with st.expander("click here to see my work >//<"):
                            st.image(st.secrets['KAGURA_VIZ'], width=500, caption="Kagura-chan's artwork") 
            else:
                st.write("Please enter a request.")


def statistic_test_tab():
    st.header('Statistic Tests')
    st.write('---')
    
    df = pd.read_excel('data.xlsx')
    # Label Encoder
    object_columns = df.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()

    for column in object_columns:
        df[column] = label_encoder.fit_transform(df[column])

    numeric = ['Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']
    scaler_power = PowerTransformer(method='yeo-johnson')
    df[numeric] = scaler_power.fit_transform(df[numeric])
    
    # Pearson Correlation Test
    target_variable_num = 'Churn Label'
    numeric_predictors = ['Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']
    correlations_num = []

    for predictor in numeric_predictors:
        correlation, p_value = pearsonr(df[predictor], df[target_variable_num])
        significance = 'Pengaruh' if abs(correlation) >= 0.5 else 'Tidak Pengaruh'
        correlation_result = {'Predictor': predictor, 'Correlation': correlation, 'P-Value': p_value, 'Significance': significance}
        correlations_num.append(correlation_result)

        correlations_df_num = pd.DataFrame(correlations_num)
        
    # Chi-Square Test
    target_variable_cat = 'Churn Label'
    categorical_predictors = ['Tenure Months', 'Location', 'Device Class', 'Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp', 'Payment Method']
    correlations = []

    for predictor in categorical_predictors:
        contingency_table = pd.crosstab(df[predictor], df[target_variable_cat])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        significance = 'Pengaruh' if p_value < 0.05 else 'Tidak Pengaruh'
        correlation_result = {'Predictor': predictor, 'Chi-Square': chi2, 'P-Value': p_value, 'Significance': significance}
        correlations.append(correlation_result)

        correlations_df_cat = pd.DataFrame(correlations)
    
    # Label Encoder Result
    st.header('Data After Label Encoding and Scaling Result')
    st.dataframe(df)

    # Pearson Correlation Test Result
    st.header('Exploring Linear Relationships with Pearson Correlation')
    
    st.image('cor1.jpeg',width=400)
    st.write("Tes korelasi Pearson digunakan untuk mengukur kekuatan dan arah hubungan linear antara dua variabel kontinu. "
         "Ini menghasilkan koefisien korelasi Pearson yang berkisar dari -1 hingga 1. "
         "Nilai 1 menunjukkan hubungan positif sempurna, nilai -1 menunjukkan hubungan negatif sempurna, "
         "dan nilai 0 menunjukkan tidak adanya hubungan linear.")

    st.write("Tipe data yang dapat digunakan dalam tes korelasi Pearson adalah data kontinu atau numerik, "
         "seperti variabel suhu, berat badan, atau pendapatan. "
         "Tes ini cocok untuk mengukur hubungan antara dua variabel yang dapat diukur dalam skala interval atau rasio. "
         "Data yang berbentuk distribusi normal atau mendekati distribusi normal akan memberikan hasil yang lebih dapat diandalkan.")

    st.dataframe(correlations_df_num)
    
    # Pearson Correlation Test Result
    st.header('Exploring Categorical Relationships with Chi-Square Test')
    
    st.image('cor2.jpg',width=400)
    
    st.write("Uji Chi-Square digunakan untuk menguji hubungan antara dua variabel kategorikal. Ini menghasilkan nilai Chi-Square, "
         "P-Value, dan menginterpretasikan signifikansinya.")

    st.write("Tipe data yang dapat digunakan dalam uji Chi-Square adalah data kategorikal, seperti jenis kelamin, status kepemilikan perangkat, "
         "atau variabel dengan kategori yang dapat dihitung.")
    st.dataframe(correlations_df_cat)
    
    # Pearson Correlation Test Result
    st.header('Exploring Ordinal Relationships with Spearman Test')
    st.image('cor3.jpg',width=400)
    
    st.write("Uji korelasi Spearman digunakan untuk mengukur hubungan statistik non-linear antara dua variabel ordinal atau interval. " 
             "Tujuannya adalah untuk mengevaluasi sejauh mana perubahan dalam satu variabel terkait dengan perubahan dalam variabel lain, meskipun hubungan tersebut mungkin tidak bersifat linier.")
    
    st.write("Hasil uji:")
    # Perform Spearman rank correlation test
    spearman_corr, p_value = spearmanr(df['Tenure Months'], df['Churn Label'])

    # Display the results in Streamlit
    st.write(f"Spearman Rank Correlation: {spearman_corr}")
    st.write(f"P-value: {p_value}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        st.write("Variabel Tenure Months memiliki korelasi yang signifikan terhadap variabel Churn Label (tolak H0)")
    else:
        st.write("Tidak terdapat korelasi yang signifikan antara variabel Tenure Months (gagal tolak H0)")
    
    
    # Normality Test
    st.header('Univariate Normality Tests')
    
    st.subheader('Shapiro-Wilk Test:')
    
        # Specify the columns you want to test for normality
    columns_to_test = ['Tenure Months', 'Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']

    # Iterate through selected columns
    for column in columns_to_test:
        # Perform Shapiro-Wilk test
        stat, p_value = shapiro(df[column])
        
        # Display the results in Streamlit
        st.subheader(f"Uji Pada Variabel' {column}'")
        st.write(f"Statistic: {stat}")
        st.write(f"P-value: {p_value}")
        
        # Check the p-value and provide interpretation
        alpha = 0.05
        if p_value > alpha:
            st.write("Data terlihat berdistribusi normal (gagal menolak H0)")
        else:
            st.write("Data tidak terlihat berdistribusi normal (menolak H0)")
        
        st.write("\n")
        
    
    # Set a custom style with no background and only the plots
    plt.style.use({'axes.facecolor': 'none', 'figure.facecolor': 'none'})

    # Specify the columns you want to visualize
    columns_to_visualize = ['Tenure Months', 'Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']

    # Set a gradient color palette
    colors = sns.color_palette("viridis", len(columns_to_visualize))

    # Set white as the text color
    text_color = 'white'

    # Iterate through selected columns
    for i, column in enumerate(columns_to_visualize):
        # Create a new Streamlit column
        col1, col2 = st.columns(2)

        # Histogram
        with col1:
            st.subheader(f'Histogram - {column}')
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sns.histplot(df[column], kde=True, color=colors[i])
            plt.title(f'Histogram - {column}', color=text_color)
            plt.xlabel('', color=text_color)  # Hide x-axis label for a cleaner look
            plt.setp(ax_hist.get_xticklabels(), color=text_color)  # Set x-axis tick labels color
            plt.setp(ax_hist.get_yticklabels(), color=text_color)  # Set y-axis tick labels color
            ax_hist.xaxis.label.set_color(text_color)  # Set x-axis label color
            ax_hist.yaxis.label.set_color(text_color)  # Set y-axis label color
            st.pyplot(fig_hist, bbox_inches='tight', pad_inches=0, use_container_width=True)
            plt.close(fig_hist)  # Close the Matplotlib figure to prevent it from being displayed again
            
        # Q-Q Plot
        with col2:
            st.subheader(f'Q-Q Plot - {column}')
            fig_qqplot, ax_qqplot = plt.subplots(figsize=(8, 4))
            qqplot(df[column], line='s', ax=ax_qqplot, color=colors[i])
            plt.title(f'Q-Q Plot - {column}', color=text_color)
            plt.xlabel('', color=text_color)  # Hide x-axis label for a cleaner look
            plt.setp(ax_qqplot.get_xticklabels(), color=text_color)  # Set x-axis tick labels color
            plt.setp(ax_qqplot.get_yticklabels(), color=text_color)  # Set y-axis tick labels color
            ax_qqplot.xaxis.label.set_color(text_color)  # Set x-axis label color
            ax_qqplot.yaxis.label.set_color(text_color)  # Set y-axis label color
            st.pyplot(fig_qqplot, bbox_inches='tight', pad_inches=0, use_container_width=True)
            plt.close(fig_qqplot)  # Close the Matplotlib figure to prevent it from being displayed again
            
    
    # ANOVA
    st.header(' One-Way Analysis of variance')

    # Example data
    churn_labels = df['Churn Label'].unique()

    # Create a list of data for each group
    groups_data = [df[df['Churn Label'] == label]['Monthly Purchase (Thou. IDR)'] for label in churn_labels]

    # Perform ANOVA
    f_statistic, p_value = f_oneway(*groups_data)

    # Display the results in Streamlit
    st.subheader("Variable: Monthly Purchase (Thou. IDR) based on Churn Label")
    st.write(f"ANOVA F-statistic: {f_statistic}")
    st.write(f"P-value: {p_value}")

    # Check the p-value and provide interpretation
    alpha = 0.05
    if p_value < alpha:
        st.write("Ada Perbedaan Mean rata-rata Monthly Purchase (Thou. IDR) antara kelompok 'Yes' dan 'No' (Tolak H0)")
    else:
        st.write("Tidak Ada Perbedaan Mean rata-rata Monthly Purchase (Thou. IDR) antara kelompok 'Yes' dan 'No' (Gagal Tolak H0)")

    # ANOVA for CLTV (Predicted Thou. IDR)
    
    # Example data
    churn_labels = df['Churn Label'].unique()

    # Create a list of data for each group
    groups_data_cltv = [df[df['Churn Label'] == label]['CLTV (Predicted Thou. IDR)'] for label in churn_labels]

    # Perform ANOVA
    f_statistic_cltv, p_value_cltv = f_oneway(*groups_data_cltv)

    # Display the results in Streamlit
    st.subheader("Variable: CLTV (Predicted Thou. IDR) based on Churn Label")
    st.write(f"ANOVA F-statistic: {f_statistic_cltv}")
    st.write(f"P-value: {p_value_cltv}")

    # Check the p-value and provide interpretation
    alpha_cltv = 0.05
    if p_value_cltv < alpha_cltv:
        st.write("Ada Perbedaan Mean rata-rata CLTV (Predicted Thou. IDR) antara kelompok 'Yes' dan 'No' (Tolak H0)")
    else:
        st.write("Tidak Ada Perbedaan Mean rata-rata CLTV (Predicted Thou. IDR) antara kelompok 'Yes' dan 'No' (Gagal Tolak H0)")


        
def model_tab():
    st.header('Model')

    # Load data
    df = pd.read_excel('data.xlsx')
    
    # Preprocess data
    df.drop(columns=['Customer ID', 'Longitude', 'Latitude','Use MyApp','Video Product'], inplace=True)
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
    X_test[numeric] = scaler_power.transform(X_test[numeric])

    clf = IsolationForest(max_samples=1500, random_state=42,contamination=.075,n_jobs=-1)
    clf.fit(X_train)
    # Predictions on the test set
    predictions = clf.predict(X_train)

    # Identify indices of outliers
    outlier_indices = np.where(predictions == -1)[0]

    # Impute outliers with the median value
    for column in X_train.columns:
        median_value = X_train[column].median()
        X_train.iloc[outlier_indices, X_train.columns.get_loc(column)] = median_value

    # Apply SMOTE
    smote = SMOTEN(random_state=42, k_neighbors=3)
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    # Train RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_estimators=1510,
        min_samples_split=66,
        min_samples_leaf=3,
        max_depth=64,
        criterion='gini',n_jobs=-1,warm_start=True
    )
    rf_classifier.fit(X_train_over, y_train_over)


    y_pred = rf_classifier.predict(X_test)
    classification_rep = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred))

    with st.expander("Click here to see the detailed information performace model with 20% test data"):
        st.text(f"```{classification_rep}```")


    st.write('---')
    st.header('Input Predict')
    
    # User Input for Prediction
    tenure_months = st.number_input('Tenure Months:',value=3)
    location = st.selectbox('Location:', df['Location'].unique(),
                            help='0 = Bandung, 1 = Jakarta')
    device_class = st.selectbox('Device Class:',
                            df['Device Class'].unique(),
                            help='0 = High End, 1 = Low End, 2 = Mid End')
    games_product = st.selectbox('Games Product:', ['Yes', 'No'])
    music_product = st.selectbox('Music Product:', ['Yes', 'No'])
    education_product = st.selectbox('Education Product:', ['Yes', 'No'])
    call_center = st.selectbox('Call Center:', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method:', df['Payment Method'].unique(),
                                help='0 = Credit, 1 = Debit, 2 = Digital Wallet, 3 = Pulsa')
    monthly_purchase = st.number_input('Monthly Purchase (Thou. IDR):',value=91.910)
    cltv = st.number_input('CLTV (Predicted Thou. IDR):',value=3511.3)

    # Preprocess user input
    input_data = pd.DataFrame({
        'Tenure Months': [tenure_months],
        'Location': [location],
        'Device Class': [device_class],
        'Games Product': [games_product],
        'Music Product': [music_product],
        'Education Product': [education_product],
        'Call Center': [call_center],
        'Payment Method': [payment_method],
        'Monthly Purchase (Thou. IDR)': [monthly_purchase],
        'CLTV (Predicted Thou. IDR)': [cltv]
    })


    # Tombol Submit
    if st.button('Submit'):
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

        if prediction_label == 'Yes':
            st.markdown(f'The predicted Churn Label is: ')
            st.success(prediction_label)
        else:
            st.markdown(f'The predicted Churn Label is: ')
            st.error(prediction_label)



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
