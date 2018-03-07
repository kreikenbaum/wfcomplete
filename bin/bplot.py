#!/usr/bin/env python
'''bokeh interactive plots'''
# uses code from stackoverflow
import seaborn as sns

from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import transform


def confusion_matrix(df):
    '''shows df-confusion matrix in bokeh'''
    df.columns.name = 'Treatment'
    df.index.name = 'Prediction'
    df = df.stack().rename("value").reset_index()

    output_file("bokeh.html")

    # You can use your own palette here
    colors = sns.color_palette("magma", 20).as_hex()

    # Had a specific mapper to map color with value
    mapper = LinearColorMapper(
        palette=colors, low=df.value.min(), high=df.value.max())

    hover = HoverTool(tooltips=[
        ("Treatment", "@Treatment"),
        ("Prediction", "@Prediction"),
        ("count", "@value")
    ])

    # Define a figure
    p = figure(
        plot_width=1500,
        plot_height=700,
        title="My plot",
        x_range=list(df.Prediction.drop_duplicates()),
        y_range=list(df.Treatment.drop_duplicates()),
        toolbar_location=None,
        tools=[hover],
        x_axis_location="above")
    # Create rectangle for heatmap
    p.rect(
        x="Prediction",
        y="Treatment",
        width=1,
        height=1,
        source=ColumnDataSource(df),
        line_color=None,
        fill_color=transform('value', mapper))
    # Add legend
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=len(colors)))
    p.add_layout(color_bar, 'right')
    show(p)
