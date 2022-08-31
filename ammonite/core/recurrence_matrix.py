#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from ..utils.fisher import fisher_information,smooth_series
from ..utils.plotting import get_labels
from ..core.rqa_res import RQA_Res

class RecurrenceMatrix:
    '''Recurrence matrix object. Used for Recurrence Quantification Analysis (RQA).
    '''
    def __init__(self,matrix,time,epsilon,m,tau,series = None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.matrix = matrix
        self.time = time
        self.epsilon = epsilon
        self.m = m
        self.tau  =tau
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label

    def laplacian_eigenmaps(self,w_size, w_incre,smooth=True,block_size=None):
        '''Function to run regime change detection workflow
        
        Parameters
        ----------
            
        w_size : int
            Window size for the fisher information 
        
        w_incre : int 
            Window increment for the fisher information 

        smooth : bool; {True, False}
            Whether or not to smooth the resulting fisher information series (recommended)

        block_size : int
            Number of time points to group together for smoothing

        Returns
        -------

        FI_series : pyleoclim.Series object
        
        '''
        W = self.matrix + 1

        D = np.zeros(W.shape)

        for i in range(len(W)):
            D[i,i] = np.sum(W[:,i])
            
        L = D - W

        _, eigvec = sp.linalg.eigh(L,D)
        
        eig_data = []

        for idx, i in enumerate(self.time):
            eig_data.append([i,eigvec[idx,1],eigvec[idx,2],eigvec[idx,3],eigvec[idx,4]])
            
        time,value = fisher_information(eig_data,w_size,w_incre)
        
        FI_series = RQA_Res(time,
                            value,
                            time_name=self.time_name,
                            time_unit=self.time_unit,
                            value_name=self.value_name,
                            value_unit=self.value_unit,
                            label=self.label,
                            m=self.m,
                            tau=self.tau,
                            eps=self.epsilon
                            )

        if smooth is True:
            if block_size is None:
                block_size = int(len(FI_series.time)/15)
                FI_series = smooth_series(FI_series,block_size)
            else:
                FI_series = smooth_series(FI_series,block_size)
        
        return FI_series

    def plot(self,figsize=(8,8),xlabel=None,ylabel=None,title=None,imshow_kwargs=None):
        '''Plotting function for recurrence matrices
        
        Parameters
        ----------
        
        figsize : tuple
            Size of figure

        xlabel : str
            Label on x axis (if time name and units are present they will be used)

        ylabel : str
            Label on y axis (if time name and units are present they will be used)

        title : str
            Title of the plot

        imshow_kwargs : dict
            Dictionary of key word arguments for the imshow method from matplotlib.axes.Axes.imshow

        See also
        --------

        matplotlib.axes.Axes.imshow
        '''

        fig, ax = plt.subplots(figsize = figsize)

        imshow_kwargs={} if imshow_kwargs is None else imshow_kwargs.copy()

        if 'cmap' not in imshow_kwargs:
            imshow_kwargs['cmap'] = 'Greys'

        if 'origin' not in imshow_kwargs:
            imshow_kwargs['origin'] = 'lower'

        if 'extent' not in imshow_kwargs:
            imshow_kwargs['extent'] = [self.time[0],self.time[-1],self.time[0],self.time[-1]]

        if xlabel is None:
            xlabel,_ = get_labels(self)
        
        if ylabel is None:
            ylabel,_ = get_labels(self)
        
        if title is None and self.label is not None:
            title = f'{self.label}'

        ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)

        ax.set_title(title)

        ax.imshow(self.matrix,**imshow_kwargs)