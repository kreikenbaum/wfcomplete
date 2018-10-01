#!/usr/bin/env python
'''bokeh interactive plots'''
# uses code from stackoverflow
import seaborn as sns

from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.transform import transform


# ((df0, _), y, yp, yd) = mplot.confusion_matrix_from_result(r, zero0=True)
# bplot.confusion_matrix(df)
def confusion_matrix(
        df, file_name='/tmp/bokeh.html', title="Confusion matrix"):
    '''shows df-confusion matrix in bokeh'''
    df = df.stack().rename("value").reset_index()

    output_file(file_name, mode="inline")

    colors = sns.color_palette("magma", 20).as_hex()
    mapper = LinearColorMapper(
        palette=colors, low=df.value.min(), high=df.value.max())

    hover = HoverTool(tooltips=[
        ("Treatment", "@Treatment"),
        ("Prediction", "@Prediction"),
        ("count", "@value")
    ])

    p = figure(
        plot_width=1500,
        plot_height=700,
        title=title,
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
    p.xaxis.major_label_orientation = "vertical"
    show(p)
    return p
