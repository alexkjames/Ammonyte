#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyleoclim as pyleo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .sampling import confidence_interval

def get_labels(obj):
    '''Function to create plotting labels from object metadata.
    
    Assumes specific object properties specific to ammonite objects
    
    Parameters
    ----------
    
    obj : ammonite.TimeEmbeddedSeries, ammonite.RecurrenceMatrix, ammonite.RecurrenceNetwork'''

    if obj.time_name is not None:
        if obj.time_unit is not None:
            xlabel = f'{obj.time_name} {obj.time_unit}'
        else:
            xlabel = f'{obj.time_name}'
    else:
        xlabel = None
    
    if obj.value_name is not None:
        if obj.value_unit is not None:
            ylabel = f'{obj.value_name} {obj.value_unit}'
        else:
            ylabel = f'{obj.value_name}'
    else:
        ylabel=None

    return xlabel, ylabel

def bootstrap_plot(series, ax=None,line_color=None,fill_color=None, ci_kwargs = None):
    '''Function for plotting rqa results with confidence bounds
    
    series : pyleoclim.Series
        Pyleoclim series object containin timeseries of the plot in question

    ax : matplotlib.axes object
        Axes to plot on, if None new plot will be generated

    line_color : str, tuple
        String or rgb tuple to use for line color
    
    fill_color : str, tuple
        String or rgb tuple to use for fill color in between lines



    See also
    --------

    ammonite.utils.sampling.confidence_interval
    '''

    if not isinstance(series,pyleo.core.Series):
        raise ValueError('This function requires a pyleoclim.Series object, please reformat and pass again.')

    ci_kwargs = {} if ci_kwargs is None else ci_kwargs.copy()

    upper,lower = confidence_interval(series,**ci_kwargs)
    
    if line_color is None:
        line_color=sns.color_palette('colorblind')[0]
    if fill_color is None:
        fill_color=sns.color_palette('colorblind')[0]
    
    #Need to find points in time where the series intersects with the lower and upper confidence boundaries
    value = series.value
    time = series.time
    
    idx_ufill = []
    idx_lfill = []

    #Need to find all points that lie above or below our confidence interval
    for idx,v in enumerate(value):
        if v >= upper:
            idx_ufill.append(idx)
        elif v <= lower:
            idx_lfill.append(idx)
    
    idx_sort = np.sort(np.concatenate((idx_ufill,idx_lfill)))
    
    breaks = []
    
    for i in range(1,len(idx_sort)):
        if idx_sort[i] != (idx_sort[i-1]+1):
            breaks.append(int(idx_sort[i-1]))
            breaks.append(int(idx_sort[i]))
    
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,8))

    series.plot(ax=ax,color=line_color)
    
    end = time[-1]
        
    ufill_values = np.zeros(len(time)) + upper
    lfill_values = np.zeros(len(time)) + lower
    
    for i in range(len(time)):
        if i in idx_ufill:
            ufill_values[i] = value[i]
        elif i in idx_lfill:
            lfill_values[i] = value[i]
                                   
    ax.fill_between(time,lower,upper,color=fill_color,alpha=.1)
    ax.fill_between(time,upper,ufill_values,color=fill_color)
    ax.fill_between(time,lower,lfill_values,color=fill_color)
        
    if 'fig' in locals():
        return fig, ax
    else:
        return ax