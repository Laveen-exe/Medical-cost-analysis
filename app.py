import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
import colorlover as cl
import helper

st.sidebar.title("Medical Cost Analysis")
data = pd.read_csv("insurance (2).csv")
st.markdown("<h1 style='text-align: left; color: #FFFFFE;'>Data analysis on Insurance Dataset</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: #FF2F48;'>The data at hand contains medical costs of people characterized by certain attributes. </h4>", unsafe_allow_html=True)
st.text('')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Dataset', 'Distribution of Data')
)

def create_charge_matrix(row):
    a=row['age']
    r=row['region']
    c=row['charges']
    charge_matrix.loc[a, r]+=c
    charge_count.loc[a, r]+=1

if user_menu == "Dataset":
    st.dataframe(data)
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    data_shape = data.shape[0]
    null_values = 0
    mean_age = round(data['age'].mean(),2)
    mean_bmi = round(data['bmi'].mean(),2)
    charges_mean = round(data['charges'].mean(),2)
    duplicates = 1

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h4 style='text-align: left; color: #FF2F48;'>Data Shape</h4>", unsafe_allow_html=True)
        st.title(data_shape)
    with col2:
        st.markdown("<h4 style='text-align: left; color: #FF2F48;'>Null Values</h4>", unsafe_allow_html=True)
        st.title(null_values)
    with col3:
        st.markdown("<h4 style='text-align: left; color: #FF2F48;'>Mean Age and Range (18-64)</h4>", unsafe_allow_html=True)
        st.title(mean_age)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h4 style='text-align: left; color: #FF2F48;'>Mean BMI (max = 53 (obese))</h4>", unsafe_allow_html=True)
        st.title(mean_bmi)
    with col2:
        st.markdown("<h4 style='text-align: left; color: #FF2F48;'>Charges mean (max = 63770.42)</h4>",
                    unsafe_allow_html=True)
        st.title(charges_mean)
    with col3:
        st.markdown("<h4 style='text-align: left; color: #FF2F48;'>No. of Duplicates</h4>",
                    unsafe_allow_html=True)
        st.title(duplicates)

    st.text("")
    st.text("")
    st.text("")
    st.text("")


    st.markdown("<h2 style='text-align: left; color: #FFFFFE;'>Graphical analysis and Exploration</h2>",
                unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.text("")
    st.markdown(
        "<h4 style='text-align: left; color: #FF2F48;'>Based on Sex </h4>",
        unsafe_allow_html=True)
    st.text("")
    st.text("")
    fig = helper.bubble('sex')
    st.plotly_chart(fig)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.markdown(
        "<h4 style='text-align: left; color: #FF2F48;'>Based on either an individual smokes or not. </h4>",
        unsafe_allow_html=True)
    st.text("")
    st.text("")
    fig = helper.bubble('smoker')
    st.plotly_chart(fig)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.markdown(
        "<h4 style='text-align: left; color: #FF2F48;'>Based on Region.</h4>",
        unsafe_allow_html=True)
    st.text("")
    st.text("")
    fig = helper.bubble('region')
    st.plotly_chart(fig)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
############################################################################################################################################################################
    ages = data.age.unique().tolist()
    regions = data.region.unique().tolist()
    charge_matrix = pd.DataFrame(data=0, index=ages, columns=regions).sort_index()
    charge_count = pd.DataFrame(data=0, index=ages, columns=regions).sort_index()
    data.apply(lambda row: create_charge_matrix(row), axis=1)

    # Calculating average charges
    charge_matrix /= charge_count

    z = []
    for i in range(len(charge_matrix)):
        z.append(charge_matrix.iloc[i].tolist())

    trace1 = go.Heatmap(
        x=charge_matrix.columns.tolist(),
        y=charge_matrix.index.tolist(),
        z=z,
        colorscale='Electric'
    )

    datat = [trace1]

    layout = go.Layout(
        title='<b>Average insurance charges by age and region<b>',
        height=800,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(
            family='Segoe UI',
            color='#ffffff'
        ),
        xaxis=dict(
            title='<b>Region<b>'
        ),
        yaxis=dict(
            title='<b>Age<b>'
        )
    )

    figure = go.Figure(data=datat, layout=layout)
    st.plotly_chart(figure)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
#############################################################################################################################################################################
    st.title("Overall Analysis")
    fig = px.scatter(data, x="age", y="charges", color="smoker", facet_col="sex", facet_row="region",width=1000, height=900)
    st.plotly_chart(fig)

if user_menu == "Distribution of Data":
    st.markdown("<h1 style='text-align: left; color: #FFFFFE;'>Distribution of Age, BMI and Charges columns</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<h4 style='text-align: left; color: #FF2F48;'>The normal distribution is the most common occurring distribution. We plot the ditribution for Age, BMI and Charges columns and find their skewness values. </h4>",
        unsafe_allow_html=True)

    x = np.random.randn(1000)
    hist_data = [x]
    group_labels = ['distplot']  # name of the dataset
    colors = ['#FF2F48']
    fig = ff.create_distplot(hist_data, group_labels, colors = colors)
    st.plotly_chart(fig)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.title("Age")
    st.markdown("<h1 style='text-align: left; color: #FF2F48;'>Box Plot</h1>",
                unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")


    fig = px.box(data, data['age'])
    st.plotly_chart(fig)

    # finding the 1st quartile
    q1 = np.quantile(data['age'], 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(data['age'], 0.75)
    med = np.median(data['age'])

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    st.markdown("<h4 style='text-align: left; color: #FFFFFE;'>The IQR, Upper Bound and Lower Bound are 24, 87 and -9</h4>",
                unsafe_allow_html=True)
    outliers = data['age'][(data['age'] <= lower_bound) | (data['age'] >= upper_bound)]
    st.markdown("<h4 style='text-align: left; color: #FF2F48;'>The following are the outliers in the boxplot : 0</h4>",
                unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>Displot (Distribution Plot) of Age</h4>",
        unsafe_allow_html=True)
    group_labels = ['distplot']  # name of the dataset
    colors = ['#FF2F48']
    fig = ff.create_distplot([data['age']], group_labels, colors=colors)
    st.plotly_chart(fig)
    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>The skewness of the distribution is 0.0547</h4>",
        unsafe_allow_html=True)
 ###################################################################################################################
    st.title("BMI (Body Mass Index)")
    st.markdown("<h1 style='text-align: left; color: #FF2F48;'>Box Plot</h1>",
                unsafe_allow_html=True)

    fig = px.box(data, data['bmi'])
    st.plotly_chart(fig)

    # finding the 1st quartile
    q1 = np.quantile(data['bmi'], 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(data['bmi'], 0.75)
    med = np.median(data['bmi'])

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>The IQR, Upper Bound and Lower Bound are 8.41, 47.31 and 13.67</h4>",
        unsafe_allow_html=True)
    outliers = data['bmi'][(data['bmi'] <= lower_bound) | (data['bmi'] >= upper_bound)]
    st.markdown("<h4 style='text-align: left; color: #FF2F48;'>The following are the outliers in the boxplot : 10</h4>",
                unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>Displot (Distribution Plot) of Age</h4>",
        unsafe_allow_html=True)
    group_labels = ['distplot']  # name of the dataset
    colors = ['#FF2F48']
    fig = ff.create_distplot([data['bmi']], group_labels, colors=colors)
    st.plotly_chart(fig)
    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>The skewness of the distribution is 0.283</h4>",
        unsafe_allow_html=True)

 ###################################################################################################################
    st.title("Charges")
    st.markdown("<h1 style='text-align: left; color: #FF2F48;'>Box Plot</h1>",
                unsafe_allow_html=True)

    fig = px.box(data, data['charges'])
    st.plotly_chart(fig)

    # finding the 1st quartile
    q1 = np.quantile(data['charges'], 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(data['charges'], 0.75)
    med = np.median(data['charges'])

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>The IQR, Upper Bound and Lower Bound are 11911.37, 34524.77 and -13120.71</h4>",
        unsafe_allow_html=True)
    outliers = data['bmi'][(data['charges'] <= lower_bound) | (data['charges'] >= upper_bound)]
    st.markdown("<h4 style='text-align: left; color: #FF2F48;'>The following are the outliers in the boxplot : 139</h4>",
                unsafe_allow_html=True)

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>Displot (Distribution Plot) of Age</h4>",
        unsafe_allow_html=True)
    group_labels = ['distplot']  # name of the dataset
    colors = ['#FF2F48']
    fig = ff.create_distplot([data['charges']], group_labels, colors=colors)
    st.plotly_chart(fig)
    st.markdown(
        "<h4 style='text-align: left; color: #FFFFFE;'>The skewness of the distribution is 1.515</h4>",
        unsafe_allow_html=True)
