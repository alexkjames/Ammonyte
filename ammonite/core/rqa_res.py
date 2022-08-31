#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyleoclim as pyleo
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from ..utils.sampling import confidence_interval

class RQA_Res(pyleo.Series):
    '''Class for storing the result of various RQA techniques'''

    def __init__(self, time, value, time_name=None, time_unit=None, value_name=None, value_unit=None, label=None, m=None,tau=None,eps=None):
        super().__init__(time,value,time_name,time_unit,value_name,value_unit,label)
        self.m = m
        self.tau = tau
        self.eps = eps

    def confidence_fill_plot(self,ax=None,line_color=None,fill_color=None,fill_alpha=None,transition_interval=None,xlabel=None,ylabel=None,marker=None,
                     markersize=None,linestyle=None,linewidth=None,alpha=None,label=None,zorder=None,plot_kwargs=None,ci_kwargs=None,
                     background_series=None,background_kwargs=None,legend=True,lgd_kwargs=None):

        '''Function for plotting rqa results with confidence bounds

        Parameters
        ----------

        ax : matplotlib.axes object
            Axes to plot on, if None new plot will be generated

        line_color : str, tuple
            String or rgb tuple to use for line color
        
        fill_color : str, tuple
            String or rgb tuple to use for fill color in between lines

        fill_alpha : float
            Transparency of the fill, default is .1

        transition_interval : list,tuple
            Upper and lower bound for the transition interval

        marker : str
            e.g., 'o' for dots
            See `matplotlib.markers <https://matplotlib.org/stable/api/markers_api.html>`_ for details

        markersize : float
            the size of the marker

        linestyle : str
            e.g., '--' for dashed line
            See `matplotlib.linestyles <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for details

        linewidth : float
            the width of the line

        alpha : float
            Transparency of the line

        label : str
            the label for the line

        xlabel : str
            the label for the x-axis

        ylabel : str
            the label for the y-axis

        title : str
            the title for the figure

        zorder : int
            The default drawing order for all lines on the plot

        plot_kwargs : dict
            Key word arguments for the main plot, see `pyleoclim.Series.plot <https://pyleoclim-util.readthedocs.io/en/latest/core/api.html#series-pyleoclim-series>_ for details

        ci_kwargs : dict
            Key word arguments for calculating the confidence interval. Only to be used if `transition_interval` is not passed. See ammonite.utils.sampling.confidence_interval for details

        background_series : pyleoclim.Series
            Optional to pass a different series that will be plotted behind the main series plot

        background_kwargs : dict
            Key word arguments for the background plot. If none are passed, the color of the main plot will be re-used 
            and alpha will be set to .2

        legend : bool; {True,False}
            Whether or not to plot the legend

        lgd_kwargs : dict

        Returns
        -------

        fig : matplotlib.figure
            The figure object from matplotlib.
            See `matplotlib.pyplot.figure <https://matplotlib.org/stable/api/figure_api.html>`_ for details.

        ax : matplotlib.axis 
            The axis object from matplotlib. 
            See `matplotlib.axes <https://matplotlib.org/stable/api/axes_api.html>`_ for details.


        See also
        --------

        ammonite.utils.sampling.confidence_interval

        '''

        ci_kwargs = {} if ci_kwargs is None else ci_kwargs.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        series = self.interp(step=1)

        if not transition_interval:
            transition_interval = confidence_interval(series,**ci_kwargs)
        
        if line_color is None:
            line_color=sns.color_palette('colorblind')[0]
        if fill_color is None:
            fill_color=sns.color_palette('colorblind')[0]
        
        #Need to find points in time where the series intersects with the lower and upper confidence boundaries
        value = series.value
        time = series.time
        
        idx_ufill = []
        idx_lfill = []

        upper = max(transition_interval)
        lower = min(transition_interval)

        #Need to find all points that lie above or below our confidence interval
        for idx,v in enumerate(value):
            if v >= upper:
                idx_ufill.append(idx)
            elif v <= lower:
                idx_lfill.append(idx)
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(12,8))

        if background_series:

            if not isinstance(background_series,pyleo.core.Series):
                raise ValueError('background_series does not appear to be a pyleoclim.Series object')
            
            #Standardize background_series values
            background_series.value -= np.mean(background_series.value)
            background_series.value /= max(background_series.value)

            #Adjust scaling to match that of main series
            background_series.value *= max(series.value - np.mean(series.value))
            background_series.value += np.mean(series.value)

            background_kwargs = {} if background_kwargs is None else background_kwargs.copy()

            if 'alpha' not in background_kwargs:
                background_kwargs['alpha'] = .2
            if 'color' not in background_kwargs:
                background_kwargs['color'] = line_color

            background_series.plot(ax=ax, **background_kwargs)

        if fill_alpha is None:
            fill_alpha = .1

        if xlabel is None:
            xlabel = f'{self.time_name} [{self.time_unit}]'

        if ylabel is None:
            ylabel = f'{self.value_name} [{self.value_unit}]'

        if label is None:
            label = self.label

        if label is not None:
            plot_kwargs.update({'label': label})

        if marker is not None:
            plot_kwargs.update({'marker': marker})

        if markersize is not None:
            plot_kwargs.update({'markersize': markersize})

        if linestyle is not None:
            plot_kwargs.update({'linestyle': linestyle})

        if linewidth is not None:
            plot_kwargs.update({'linewidth': linewidth})

        if alpha is not None:
            plot_kwargs.update({'alpha': alpha})

        if zorder is not None:
            plot_kwargs.update({'zorder': zorder})

        series.plot(ax=ax,color=line_color,xlabel=xlabel,ylabel=ylabel,plot_kwargs=plot_kwargs,lgd_kwargs=lgd_kwargs,legend=legend)
            
        ufill_values = np.zeros(len(time)) + upper
        lfill_values = np.zeros(len(time)) + lower
        
        for i in range(len(time)):
            if i in idx_ufill:
                ufill_values[i] = value[i]
            elif i in idx_lfill:
                lfill_values[i] = value[i]
                                    
        ax.fill_between(time,lower,upper,color=fill_color,alpha=fill_alpha)
        ax.fill_between(time,upper,ufill_values,color=fill_color)
        ax.fill_between(time,lower,lfill_values,color=fill_color)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax

    def confidence_scatter_plot(self,ax=None,transition_interval=None,marker=None,color=None,size=None,
                    label=None,xlabel=None,ylabel=None,title=None,scatter_kwargs=None,ci_kwargs=None,
                    background_series=None,background_kwargs=None,legend=True,lgd_kwargs=None):

        '''Function for plotting rqa results with confidence bounds

        Parameters
        ----------

        ax : matplotlib.axes object
            Axes to plot on, if None new plot will be generated

        transition_interval : list,tuple
            Upper and lower bound for the transition interval

        marker : str
            e.g., 'o' for dots
            See `matplotlib.markers <https://matplotlib.org/stable/api/markers_api.html>`_ for details

        size : float
            the size of the marker

        color : str, tuple
            String or rgb tuple to use for marker color

        label : str
            the label for the scatter

        xlabel : str
            the label for the x-axis

        ylabel : str
            the label for the y-axis

        title : str
            the title for the figure

        scatter_kwargs : dict
            Key word arguments for the main plot, see `matplotlib.pyplot.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>_ for details

        ci_kwargs : dict
            Key word arguments for calculating the confidence interval. Only to be used if `transition_interval` is not passed. See ammonite.utils.sampling.confidence_interval for details

        background_series : pyleoclim.Series
            Optional to pass a different series that will be plotted behind the main series plot

        background_kwargs : dict
            Key word arguments for the background plot. If none are passed, the color of the main plot will be re-used 
            and alpha will be set to .2

        legend : bool; {True,False}
            Whether or not to plot the legend

        lgd_kwargs : dict

        Returns
        -------

        fig : matplotlib.figure
            The figure object from matplotlib.
            See `matplotlib.pyplot.figure <https://matplotlib.org/stable/api/figure_api.html>`_ for details.

        ax : matplotlib.axis 
            The axis object from matplotlib. 
            See `matplotlib.axes <https://matplotlib.org/stable/api/axes_api.html>`_ for details.


        See also
        --------

        ammonite.utils.sampling.confidence_interval

        '''

        ci_kwargs = {} if ci_kwargs is None else ci_kwargs.copy()
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if not transition_interval:
            transition_interval = confidence_interval(self,**ci_kwargs)
        
        idx_fill = []

        upper = max(transition_interval)
        lower = min(transition_interval)

        #Need to find all points that lie above or below our confidence interval
        for idx,v in enumerate(self.value):
            if v >= upper or v <= lower:
                idx_fill.append(idx)
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(12,8))

        if background_series:

            if not isinstance(background_series,pyleo.core.Series):
                raise ValueError('background_series does not appear to be a pyleoclim.Series object')
            
            #Standardize background_series values
            background_series.value -= np.mean(background_series.value)
            background_series.value /= max(background_series.value)

            #Adjust scaling to match that of main series
            background_series.value *= max(self.value - np.mean(self.value))
            background_series.value += np.mean(self.value)

            background_kwargs = {} if background_kwargs is None else background_kwargs.copy()

            if 'alpha' not in background_kwargs:
                background_kwargs['alpha'] = .2

            background_series.plot(ax=ax, **background_kwargs)

        if xlabel is None:
            xlabel = f'{self.time_name} [{self.time_unit}]'

        if ylabel is None:
            ylabel = f'{self.value_name} [{self.value_unit}]'

        if label is None:
            label = self.label

        if marker is None:
            marker = 'o'
        
        if color is None:
            color = sns.color_palette('colorblind')[0]

        if size is None:
            size = 1

        ax.scatter(self.time[idx_fill],self.value[idx_fill],label=label,marker=marker,s=size,c=color,
                    **scatter_kwargs)

        ax.axhline(max(transition_interval),min(self.time),max(self.time))
        ax.axhline(min(transition_interval),min(self.time),max(self.time))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if legend:
            ax.legend()

        if 'fig' in locals():
            return fig, ax
        else:
            return ax

    # def ci_scatter_plot(self):
    #     '''Function to create a scatter plot with a confidence interval
        
    #     Parameters
    #     ----------
    #     '''
    #     fig,ax = bootstrap_scatter_plot(self)

    #     return fig, ax