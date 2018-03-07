#!/usr/bin/env python
'''bokeh interactive plots'''
# uses code from stackoverflow
from bokeh.plotting import figure, show  # , output_file


def confusion_matrix(confmat_df, normalize=True):
    '''shows confusion matrix in bokeh'''
    if normalize:
        confmat_df = confmat_df / confmat_df.sum(axis=1)
    p = figure(title="Confusion Matrix",
               x_axis_location="above", tools="hover,save",
               x_range=list(reversed(confmat_df.columns)),
               y_range=confmat_df.index)
    # x_range=list(reversed(domainnames)), y_range=domainnames)
    p.rect('index', 'columns', source=confmat_df, hover_line_color='black')
