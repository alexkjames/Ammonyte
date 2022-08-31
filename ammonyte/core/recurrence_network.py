#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..core.recurrence_matrix import RecurrenceMatrix

class RecurrenceNetwork(RecurrenceMatrix):
    '''Recurrence network object. Used for Recurrence Network Analysis (RNA).

    Child of ammonyte.RecurrenceMatrix, so it has all of the RecurrenceMatrix methods
    plus additional methods defined here
    '''
    def __init__(self,matrix,time,epsilon,series=None,value_name=None,value_unit=None,time_name=None,time_unit=None,label=None):
        self.matrix = matrix
        self.time = time
        self.epsilon = epsilon
        self.series = series
        self.value_name = value_name
        self.value_unit = value_unit
        self.time_name = time_name
        self.time_unit = time_unit
        self.label = label