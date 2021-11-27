import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import colorlover as cl
colors=cl.scales['12']['qual']
chosen_colors=[j for i in colors for j in colors[i]]

data = pd.read_csv('insurance (2).csv')

def bubble(trace_col):
    data_ = []
    trace_vals = list(data[trace_col].unique())
    for i in range(len(trace_vals)):
        data_.append(
            go.Scatter(
                x=data[data[trace_col] == trace_vals[i]].age,
                y=data[data[trace_col] == trace_vals[i]].bmi,
                mode='markers',
                marker=dict(
                    color=chosen_colors[i * 4 + 1],
                    opacity=0.5,
                    size=data[data[trace_col] == trace_vals[i]].charges / 3000,
                    line=dict(
                        width=0.0
                    )
                ),
                text='Age:' + data[data[trace_col] == trace_vals[i]].age.astype(str) + '<br>' + 'BMI:' +
                     data[data[trace_col] == trace_vals[i]].bmi.astype(str) + '<br>' + 'Charges:' + data[
                         data[trace_col] == trace_vals[i]].charges.astype(str) + '<br>' + trace_col + ':' +
                     trace_vals[i],
                hoverinfo='text',
                name=trace_vals[i]
            )
        )

    layout = go.Layout(
        title='<b>Insurance cost comparison for different ages / BMIs</b>',
        hovermode='closest',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(
            family='Segoe UI',
            color='#ffffff'
        ),
        xaxis=dict(
            title='<b>Age<b>'
        ),
        yaxis=dict(
            title='<b>BMI<b>'
        ),
        legend=dict(
            orientation='h',
            x=0,
            y=1.1
        ),
        shapes=[
            dict(
                type='rect',
                xref='x',
                x0=min(data.age) - 1,
                x1=max(data.age) + 1,
                yref='y',
                y0=18.5,
                y1=24.9,
                line=dict(
                    width=0.0
                ),
                fillcolor='rgba(255,255,255,0.2)'
            )
        ],
        annotations=[
            dict(
                xref='x',
                x=45,
                yref='y',
                y=18.5,
                text='Healthy BMI zone',
                ay=35
            )
        ]
    )

    figure = go.Figure(data=data_, layout=layout)
    return figure

