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
from openai import OpenAI as ai
import time

#Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
import numpy as np
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
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
    # st.header("Exploratory Data Analysis")
    # st.write('---')
    data = pd.read_excel('data.xlsx')

    # st.subheader("DataFrame Info")
    # nrows = st.number_input("Show number of rows", min_value=1, value=5, step=1)
    # st.dataframe(data.head(nrows), use_container_width=True)
    
    # # Capture the .info() output into a string
    # buffer = io.StringIO()
    # data.info(buf=buffer)
    # info_text = buffer.getvalue()
    # buffer.close()
    
    # # Display .info() of the DataFrame as Markdown
    # with st.expander("Click here to see the detailed information"):
    #     st.markdown(f"```{info_text}```", unsafe_allow_html=True)

    with st.container():

        col1, col2 = st.columns([0.6,0.4])
        with col1:
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
            m = folium.Map(location=map_center, zoom_start=9, width='100%', height='100%', tiles='OpenStreetMap')

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

            
            # st.write("This map visualizes churn rates by hexagon.")
            st.write("Green represents lower churn rates, while red indicates higher churn rates.")
            components.html(m._repr_html_(), height=500)
            
        with col2:
            st.subheader("General Churn Rate")
            fig = px.pie(data.groupby('Churn Label')['Customer ID'].nunique().reset_index(), 
                values='Customer ID', 
                names='Churn Label')
            fig.update_layout(width=500,height=370)
            st.plotly_chart(fig)
            with st.expander("**Problem Statement**"):
                st.write("Sebagian besar pelanggan tetap setia (73.5%), sementara hanya sebagian kecil yang berhenti (26,5%). Tantangannya, bagaimana meminimalisir agar yang berhenti dapat tetap setia")

    with st.container():

        col3, col4, col5 = st.columns([0.45,0.275,0.275])
        with col3:
            st.subheader("Customer's lifetime in the service")
            fig = px.histogram(data, x="Tenure Months", color="Churn Label",marginal="box" )
            fig.update_layout(height=500)
            st.plotly_chart(fig)

        with col4:
            st.subheader("What type of clients' devices who left the service?")
            fig = px.pie(
                data.groupby(['Device Class', 'Churn Label'])['Customer ID'].count().reset_index(),
                values='Customer ID',
                facet_col='Churn Label',
                names='Device Class',
                facet_col_wrap=1,hole=.5
            )
            fig.update_layout(width=350,height=500)
            st.plotly_chart(fig)
        
        with col5:
            st.subheader("What type of clients' payment method who left the service?")
            fig = px.pie(data.groupby(['Payment Method','Churn Label'])['Customer ID'].count().reset_index(), 
                    values='Customer ID', 
                    facet_col = 'Churn Label',
                    names='Payment Method',
                    hole = .5,facet_col_wrap=1
                    )
            fig.update_layout(
                width=350, height=500
            )
            st.plotly_chart(fig)
    with st.container():
        col6, col9 = st.columns([0.9999,0.0001])
        with col6:
            st.subheader('Churn rate by customer payment method')
            fig = px.pie(data.groupby(['Payment Method','Churn Label'])['Customer ID'].count().reset_index(), 
                    values='Customer ID', 
                    names='Churn Label',
                    facet_col = 'Payment Method',
                    color = 'Churn Label',
                    )
            fig.update_layout(
                width=1200,  # Adjust the width
                # height=800,  # Adjust the height
                margin=dict(l=50, r=50, b=50, t=50, pad=4),
            )
            st.plotly_chart(fig)
        

    with st.container():
        col7, col8 = st.columns([.5 , .5])
        with col7:
            st.subheader("Correlation between Feature")
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

            fig.update_layout(
                height=600,  # Adjust the height as needed
            #     width=500,   # Adjust the width as needed
            )
            # Display the Plotly heatmap using st.plotly_chart
            st.plotly_chart(fig)

        with col8:
            st.subheader("Detail Information From Heatmap")
            fig = px.bar(df_dummies.corr()['Churn Label'].sort_values(ascending = False), 
             color = 'value')
            
            fig.update_layout(
                # width=1100,  # Adjust the width
                height=600,  # Adjust the height
            )
            st.plotly_chart(fig)


    
    # with st.expander("Click here to see the 'Churn Label' statistics"):
    #     st.subheader("Quantiles of Tenure Months")
    #     quantiles = data.groupby('Churn Label')['Tenure Months'].quantile([0.50, 0.75, 0.90, 0.95])
    #     st.write(quantiles)

    #     st.subheader("Mean of Tenure Months")
    #     means = data.groupby('Churn Label')['Tenure Months'].mean()
    #     st.write(means)

def chat_tab():
    # plt.use('TkAgg')

    # kagu1, kagu2, kagu3 = st.columns([1,5,1])

    # with kagu2:
    #     st.header("Hello, I'm Kagura-chan!")

    st.markdown("""
        <style>
        .header-style {
        font-size: 36px;
        text-align: center;
        font-weight: bold;

        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="header-style">Hello, I\'m Kagura-chan!</p><br>', unsafe_allow_html=True)


    API_KEY = st.secrets['OPENAI_API_KEY']
    llm = OpenAI(api_token=API_KEY)

    df = pd.read_excel('data.xlsx')
    df = SmartDataframe(df, config={'llm':llm})
    
    # Creating two columns for the layout
    col1, col2 = st.columns([2, 3])  # Adjust the ratio as needed

    keyword = ['viz', 'visualization', 'plot', 'barplot', 'graf', 'graph', 'visualisasi', 'chart', 'gambar']

    # Column 1 for the image
    with col1:
        st.image('kagurachan.png', width=400)

    # Column 2 for the user prompt
    with col2:
        st.write("""
Kagura-chan adalah salah satu bot dari **KAGURABACHI** yang dapat kamu anggap seperti asisten pribadi. 
Ia bisa memahami dataset yang ada dan ia juga bisa mengerjakan pekerjaan mudah seperti membuat **visualisasi**. 
Jangan memberi perintah yang terlalu susah karena bisa saja Kagura-chan malah **membencimu**.

## Fitur Utama
- **Asisten Pribadi yang Ramah dan Cerdas**: Kagura-chan didesain untuk memahami kebutuhan Anda. Dari menjawab pertanyaan seputar dataset hingga membantu Anda dalam analisis data, Kagura-chan adalah partner yang Anda butuhkan untuk menjelajah dunia data yang kompleks.
- **Ahli Visualisasi Data**: Minta saja dan Kagura-chan akan mengubah data mentah menjadi visualisasi yang mudah dipahami dan menarik. Grafik, diagram, dan peta interaktif? Semua menjadi lebih mudah dengan sentuhan Kagura-chan.
""") 
        
    
    r1, r2 = st.columns([2, 3])
    
    with r1:
        with st.form("prompt_area"):
            prompt = st.text_input("Ask me anything ^ ^")
            submitted = st.form_submit_button("Ask Kagura-chan")

    with r2:
        # Logic for processing the prompt and displaying results
        if submitted and prompt:
            with st.spinner("Kagura-chan is thinking (♡μ_μ), please wait..."):
                bot_out = df.chat(prompt)
                st.write(bot_out if bot_out is not None else '')

                # Additional logic for visualization
                if any(kw in prompt.lower() for kw in keyword):
                    with st.expander("click here to see my work >//<"):
                        st.image(st.secrets['KAGURA_VIZ'], width=500, caption="Kagura-chan's artwork")
        elif submitted:
            st.write("Please enter a request.")

    st.write('---')


    st.markdown("""
        <style>
        .header-style {
        font-size: 36px;
        text-align: center;
        font-weight: bold;

        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="header-style">Yo! I\'m Bachi</p><br>', unsafe_allow_html=True)

    # Creating two columns for the layout
    bac1, bac2 = st.columns([3,2])  # Adjust the ratio as needed

    # Column 1 for the bachi desc
    with bac1:
        st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            Rekan terpercaya Kagura-chan dari keluarga KAGURABACHI, Bachi dirancang khusus untuk menjadi kekuatan pendorong di balik strategi pemasaran Anda. Bukan hanya chatbot biasa; Bachi adalah asisten cerdas yang berfokus pada analisis data untuk meningkatkan efektivitas strategi pemasaran Anda.

            ## Mengapa Bachi Menjadi Pilihan Ideal untuk Strategi Pemasaran?

            - **Ahli dalam Menentukan Promo yang Tepat:** Bachi memahami kebutuhan unik setiap pelanggan. Dengan kecerdasan buatan canggih, ia dapat menganalisis tren dan data pelanggan untuk menentukan jenis promosi yang paling efektif, membantu meningkatkan loyalitas pelanggan terhadap perusahaan Anda.

            - **Data-Driven Insights:** Berbekal dengan kemampuan analisis data yang kuat, Bachi menyediakan wawasan yang didasarkan pada data untuk membantu Data Analyst dalam merancang strategi pemasaran yang lebih terarah dan efektif.
            </div>
        """, unsafe_allow_html=True)


    # Column 2 for the bachi image
    with bac2:
        st.image('bachi.png', width=450)


    # def load_rules(ruler):
    #     """Load and parse association rules from the given file."""
    #     with open(ruler, 'r') as file:
    #         rules = [line.strip().split(';') for line in file.readlines()]
    #     return {antecedent.strip(): consequent.strip() for antecedent, consequent in rules}

    # def list_antecedents(rules):
    #     pass

    # def list_consequents(rules, antecedent):
    #     pass

    
    # available_antecedents = []
    # available_consequents = []

    antecedents_consequents_pair = {}
    consequents_list = []
    antecedents_list = []
    consequents = 0
    antecedents = 0

    def pairing_antecedents_consequents(rules):
        # antecedents_consequents_pair = {}
        
        # Split the input string into rows
        rows = rules.strip().split('\n')

        for row in rows:
            # Split each row into two parts based on the ';' character
            parts = row.split(';')
        
            # Assign consequents and antecedents
            consequents = parts[0].strip()
            antecedents = parts[1].strip()

            if antecedents in antecedents_consequents_pair:
                # If it exists, append the new value (consequents) to the existing list
                antecedents_consequents_pair[antecedents].append(consequents)
            else:
                # If the key doesn't exist, create a new entry with the key and a list containing the value
                antecedents_consequents_pair[antecedents] = [consequents]

        return antecedents_consequents_pair

    with open('rules.txt', 'r') as file:
        # Read the content of the file
        rules = file.read()  
    pairing_antecedents_consequents(rules)


    # Define your list of options
    options1 = [
    'Customer yang tidak bisa menggunakan MyApp karena tidak memiliki sinyal',
    'Customer dengan Device Low-End dan tidak menggunakan Call Center',
    'Customer yang tidak menggunakan MyApp karena tidak memiliki sinyal tidak menggunakan Call Center',
    'Customer yang menggunakan produk video',
    'Customer yang memang tidak ingin menggunakan MyApp',
    'Customer yang menggunakan MyApp',
    'Customer yang memang tidak ingin menggunakan produk video',
    'Customer yang memiliki Device High-End',
    'Gamers',
    ]

    options2 = [
        'Steady Low-to-Mid Spenders (~ IDR 94,580)',
        'Semi-Consistent Moderate Spenders (~ IDR 95,120)',
        'Consistent Moderate Spenders (~ IDR 95,310)',
        'Consistently High Spenders (~ IDR 95,530)',
        'Engaged High Spenders (~ IDR 97,690)',
        'Potential Value Seekers (~ IDR 101,860)'
    ]

    rekom1, rekom2 = st.columns([2,2]) 

    with rekom1:
        # select
        selected_behavior = st.selectbox('Pilih behavior customer kamu', options1, index = 8)
    
    with rekom2:
        selected_cluster = st.radio('Pilih segmentasi pasar yang kamu targetkan',options=options2)

    # st.write('pairs:',antecedents_consequents_pair)
    knowledge = """
    Device Class (Device classification)
    Games Product (Whether the customer uses the internet service for games product)
    Music Product (Whether the customer uses the internet service for music product)
    Education Product (Whether the customer uses the internet service for education product)
    Call Center (Whether the customer uses the call center service)
    Video Product (Whether the customer uses video product service)
    Use MyApp (Whether the customer uses MyApp service)
    Payment Method (The method used for paying the bill)
    Monthly Purchase (Total customer’s monthly spent for all services with the unit of thousands of IDR)
    Churn Label (Whether the customer left the company in this quarter)
    """
    template = f"""
    Kamu adalah seorang Sales Marketing Strategist yang handal dalam menangani customer churn menggunakan pengetahuan dasar ini: {knowledge}.
    Kasus kali ini memiliki perilaku customer {selected_behavior} dengan target pasar {selected_cluster} (IDR yang dimaksud disini adalah rata-rata pengeluaran cluster tersebut untuk perbulannya).
    Buatkan aku sebuah promosi atau kebijakan bisnis yang bisa digunakan agar customer tersebut tidak churn !
    Berikan aku SATU saja !
    """
    bachiprompt = template

    client = ai(
    api_key=st.secrets['OPENAI_API_KEY'],
    )

    bachiii = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": bachiprompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # A button for users to submit their selections
    bachirekom = st.button("Ask Bachi's recommendation")
    if bachirekom:
        with st.spinner("Bachi is analyzing your request (⌐■_■), please hold on for a sec..."):
            if bachiii.choices:
                bac_out = bachiii.choices[0].message.content
            else:
                bac_out = None
            # Delay for demonstration purposes
            time.sleep(2)  # Simulate a time-consuming operation
            
        st.write(bac_out if bac_out is not None else '')

    elif bachirekom:
        st.write("Please enter a request.")

    




def statistic_test_tab():
    st.header('Statistic Tests')
    st.write('---')
    
    with st.container():
        # Creating two columns for the layout
        col1, col2 = st.columns([0.55, 0.45])  # Adjust the ratio as needed
            # Column 1 for the image
        with col1:
            st.image('detective2.png',width=420)

        # Column 2 for the user prompt
        with col2:
            st.write(' **Uji statistik** dalam dunia data seperti kaca pembesar yang membantu Data Analyst. Tanpa uji statistik, kita hanya akan menebak-nebak makna data tanpa dasar yang kuat. Dengan uji statistik ini, kita dapat **membedakan antara kebetulan dan pola yang nyata**, serta memastikan bahwa kesimpulan yang diambil didukung oleh **bukti yang kuat**. Sama halnya dengan detektif yang memilah bukti penting dari yang tidak, uji statistik membantu kita mengumpulkan dan menyusun cerita yang jelas dari data yang ada.')
    
    df = pd.read_excel('data.xlsx')
    # Label Encoder
    object_columns = df.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()

    for column in object_columns:
        df[column] = label_encoder.fit_transform(df[column])

    numeric = ['Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']
    scaler_power = PowerTransformer(method='yeo-johnson')
    df[numeric] = scaler_power.fit_transform(df[numeric])
    
    # variables
    target_variable_num = 'Churn Label'
    numeric_predictors = ['Monthly Purchase (Thou. IDR)', 'CLTV (Predicted Thou. IDR)']
    correlations_num = []
        
    # Chi-Square Test
    target_variable_cat = 'Churn Label'
    categorical_predictors = ['Location', 'Device Class', 'Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp', 'Payment Method']
    correlations = []

    for predictor in categorical_predictors:
        contingency_table = pd.crosstab(df[predictor], df[target_variable_cat])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        significance = 'Pengaruh' if p_value < 0.05 else 'Tidak Ada Pengaruh'
        correlation_result = {'Predictor': predictor, 'Chi-Square': chi2, 'P-Value': p_value, 'Significance': significance}
        correlations.append(correlation_result)

        correlations_df_cat = pd.DataFrame(correlations)
    
    # Label Encoder Result
    st.header('Data After Label Encoding and Scaling Result')
    st.dataframe(df)
    
    # Chi-Square Correlation Test Result
    st.header('Exploring Categorical Relationships with Chi-Square Test')
    
    st.image('cor2.jpg',width=400)
    
    st.write("Chi-square adalah sebuah alat statistik yang membantu menentukan apakah ada hubungan yang signifikan antara dua variabel atau hanya kebetulan semata. Mirip dengan peran seorang detektif, Chi-square membantu dalam mengungkap misteri hubungan antar variabel tersebut, memberikan pemahaman apakah pola yang terlihat dalam data itu nyata secara statistik, serta mengubahnya menjadi narasi yang bisa dipahami secara luas. Dengan Chi-square, kita dapat dengan yakin menceritakan cerita yang didukung oleh bukti statistik yang kuat, menjelaskan data dengan keyakinan, dan menghindari kesimpulan yang hanya berdasarkan asumsi semata.")
    st.dataframe(correlations_df_cat)    
    
    # ANOVA
    st.header(' One-Way Analysis of Variance')
    st.image('cor3.jpg',width=400)
    st.write("One-way ANOVA adalah alat statistik yang membantu kita mengetahui apakah terdapat perbedaan signifikan antara rata-rata dari tiga atau lebih kelompok dalam dataset kita. Ini seperti detektif yang membantu mengungkap apakah ada perbedaan yang signifikan antara kelompok-kelompok, memungkinkan kita menemukan pola atau perbedaan yang mungkin terlewatkan jika hanya melihat data secara kasar. Misalnya, dalam dataset kelompok pengguna aplikasi, ANOVA akan membantu kita memahami apakah ada perbedaan yang signifikan antara performa atau preferensi di antara kelompok-kelompok tersebut.")
    # Defining continuous target variable and categorical predictors
    target_variable = 'CLTV (Predicted Thou. IDR)'  # CLTV as continuous target variable
    categorical_predictors = ['Location', 'Device Class', 'Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp', 'Payment Method']

    # Initialize a list to store ANOVA results for each predictor
    anova_results = []

    # Loop through each categorical predictor
    for predictor in categorical_predictors:
        # Group CLTV by categories in the predictor variable
        groups = [df[df[predictor] == category][target_variable] for category in df[predictor].unique()]

        # Perform ANOVA for the predictor variable
        f_statistic, p_value = f_oneway(*groups)

        # Determine significance
        if p_value < 0.05:
            significance = 'Ada Perbedaan Signifikan Antar Mean Group'
        else:
            significance = 'Tidak Ada Perbedaan Signifikan Antar Mean Group'

        # Store ANOVA results for the predictor
        anova_result = {
            'Predictor': predictor,
            'F-Statistic': f_statistic,
            'P-Value': p_value,
            'Difference': significance
        }
        anova_results.append(anova_result)

    # Create a DataFrame from ANOVA results
    anova_df = pd.DataFrame(anova_results)

    # Display ANOVA results
    st.dataframe(anova_df)
    
    
    # Post-Hoc Test
    df.rename(columns={
    'Tenure Months': 'Tenure_Months',
    'Monthly Purchase (Thou. IDR)': 'Monthly_Purchase',
    'CLTV (Predicted Thou. IDR)': 'CLTV',
    'Device Class': 'Device_Class',
    'Games Product': 'Games_Product',
    'Music Product': 'Music_Product',
    'Education Product': 'Education_Product',
    'Call Center': 'Call_Center',
    'Video Product': 'Video_Product',
    'Use MyApp': 'Use_MyApp',
    'Payment Method': 'Payment_Method',
    'Churn Label': 'Churn_Label'

    }, inplace=True)
    
    st.header('Exploring Significances of Groups within Categorical variables')
    
    st.image('phoc.jpg',width=400)

    st.write("Post-hoc test, dalam konteks statistik, merupakan langkah lanjutan setelah analisis ANOVA untuk mengeksplorasi perbedaan di antara beberapa kelompok yang telah menunjukkan perbedaan signifikan. Ini seperti detektif yang melakukan penyelidikan lebih lanjut setelah menemukan perbedaan antar kelompok dalam ANOVA. Misalnya, jika ANOVA menunjukkan adanya perbedaan signifikan antara beberapa kelompok dalam sebuah studi, post-hoc test akan membantu mengidentifikasi pasangan kelompok mana yang secara khusus berbeda satu sama lain. Post-hoc test memberikan wawasan lebih mendalam, membantu memahami lebih jauh mengenai perbedaan spesifik antar kelompok yang tidak bisa ditentukan hanya dengan ANOVA saja.")
    
    st.subheader("Games Product")
    # Perform ANOVA 1
    model = ols('CLTV ~ Games_Product', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Games_Product'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df

    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            - **Kelompok 'No' dan 'No Internet Service':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 1 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No' dan 'Yes':** erdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 2 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No Internet Service' dan 'Yes':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 2 memiliki dampak yang signifikan pada hasil.
            
            Dengan demikian, hasil post-hoc menunjukkan bahwa ketiga kelompok tersebut memiliki perbedaan yang signifikan dalam variabel Games Product terkait dengan hasil yang diamati.
            </div>
        """, unsafe_allow_html=True)

    
    st.subheader("Music Product")
    # Perform ANOVA 2
    model = ols('CLTV ~ Music_Product', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Music_Product'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df
    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            - **Kelompok 'No' dan 'No Internet Service':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 1 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No' dan 'Yes':** erdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 2 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No Internet Service' dan 'Yes':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 2 memiliki dampak yang signifikan pada hasil.
            
            Dengan demikian, hasil post-hoc menunjukkan bahwa ketiga kelompok tersebut memiliki perbedaan yang signifikan dalam variabel Music Product terkait dengan hasil yang diamati.
            </div>
        """, unsafe_allow_html=True)
    # Perform ANOVA 3

    st.subheader("Education Product")
    # Perform ANOVA 2
    model = ols('CLTV ~ Education_Product', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Education_Product'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df
    
    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            - **Kelompok 'No' dan 'No Internet Service':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 1 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No' dan 'Yes':** erdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 2 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No Internet Service' dan 'Yes':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 2 memiliki dampak yang signifikan pada hasil.
            
            Dengan demikian, hasil post-hoc menunjukkan bahwa ketiga kelompok tersebut memiliki perbedaan yang signifikan dalam variabel Education Product terkait dengan hasil yang diamati.
            </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Call Center")
    # perform ANOVA 4
    model = ols('CLTV ~ Call_Center', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Call_Center'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df
    
    
    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            - **Kelompok 'No' dan 'Yes':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 1 memiliki dampak yang signifikan pada hasil.
            
            Dengan demikian, hasil post-hoc menunjukkan bahwa ketiga kelompok tersebut memiliki perbedaan yang signifikan dalam variabel Call Center terkait dengan hasil yang diamati.
            </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Video Product")
    # perform ANOVA 5
    
    
    model = ols('CLTV ~ Video_Product', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Video_Product'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df
    
    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            Hasil dari analisis post-hoc pada variabel Payment Method menunjukkan perbedaan signifikan antara beberapa kelompok:
            
            - **Kelompok 'No' dan 'No Internet Service':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 1 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No' dan 'Yes':** erdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 2 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No Internet Service' dan 'Yes':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 2 memiliki dampak yang signifikan pada hasil.
            
             Dengan demikian, hasil post-hoc menunjukkan bahwa ketiga kelompok tersebut memiliki perbedaan yang signifikan dalam variabel Video Product terkait dengan hasil yang diamati.
            </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Use MyApp")
    # perform ANOVA 6
    model = ols('CLTV ~ Use_MyApp', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Use_MyApp'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df
    
    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            - **Kelompok 'No' dan 'No Internet Service':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 1 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No' dan 'Yes':** erdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 2 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'No Internet Service' dan 'Yes':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 2 memiliki dampak yang signifikan pada hasil.
            
            Dengan demikian, hasil post-hoc menunjukkan bahwa ketiga kelompok tersebut memiliki perbedaan yang signifikan dalam variabel Use MyApp terkait dengan hasil yang diamati.
            </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Payment Method")
    # perform ANOVA 7
    
    model = ols('CLTV ~ Payment_Method', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD test for multiple comparisons
    tukey_results = sm.stats.multicomp.pairwise_tukeyhsd(df['CLTV'], df['Payment_Method'])

    # Convert Tukey's test results to DataFrame
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    # Add a column for significance based on a significance level of 0.05
    tukey_df['Significance'] = tukey_df['p-adj'].apply(lambda x: 'Ada Perbedaan Signifikan Antar Mean Group' if x < 0.05 else 'Tidak Ada Perbedaan Signifikan Antar Mean Group')

    # Display the Tukey's test results including significance column
    tukey_df
    
    with st.expander("For More Insights"):
            st.markdown("""
            <style>
            .justify {
                text-align: justify;
            }
            </style>
            <div class="justify">
            
            - **Kelompok 'Credit' dan 'Digital Wallet':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 2 memiliki dampak yang signifikan pada hasil yang diamati.
            
            - **Kelompok 'Credit' dan 'Pulsa':** erdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 0 dan 3 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'Debit' dan 'Digital Wallet':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 2 memiliki dampak yang signifikan pada hasil.
            
            - **Kelompok 'Debit' dan 'Pulsa':** Terdapat perbedaan signifikan antara mean group. Perbedaan antara kelompok 1 dan 3 memiliki dampak yang signifikan pada hasil.
            
            Namun, **tidak ada perbedaan signifikan** antara kelompok **'Credit' (0) dan 'Debit'(1)**, serta antara kelompok **'Digital Wallet'(2) dan 'Pulsa(3)**.
            </div>
        """, unsafe_allow_html=True)
    
    
    
        

        
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
