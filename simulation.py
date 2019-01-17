#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 10:36:05 2019

@candidatenumber: 1032562
"""

import numpy as np
from copy import deepcopy

def generate_service_time(shape, scale=1):
    """ Generates 1 realisation of a exponential-distributed random 
    variable. This counts as the duration for a customer service. """
    
    return  np.random.gamma(shape, scale, 1)

class Customer:
    
    def __init__(self, shape, scale, arrival_time, service_start_time=None,
                 service_end_time=None, wait=None):
        
        self.arrival_time = arrival_time
        self.service_start_time =  service_start_time
        self.service_time = generate_service_time(shape, scale)
        self.service_end_time = service_start_time
        self.wait = wait

    def print_details(self):
        """ Prints the details of the node. """
        
        print("arrival_time:", self.arrival_time)
        print("service_start_time:", self.service_start_time)
        print("service_time:", self.service_time)
        print("service_end_time:", self.service_end_time)
        print("wait:", self.wait)
        
def Simulate(node, rates, shapeGamma, scaleGamma, times, shiftMatrix, 
             tau=9, thres=0.1, nbOccurences=10, interval_length=20): 
    """ Simulates operations over the planning horizon with specified shift 
    vector (repeats 'nbOccurences' times). Returns True if the quality service
    constraints is satisfied. Otherwise, returns False and the index of the 
    first staffing period at which the constraint has been violated. """
    
    waiting_times_summary = {}
    # Simulate 'nbOccurences' times
    for occurence in range(nbOccurences):
        poissonParams = {}
        Customers=[]
        count_per_interval = {}
        
        # Set workforce for each staffing period
        for time in times:
            n = int(time/interval_length)
            count_per_interval[str(time)] = sum(shiftMatrix[:,n]*node.shift_vector)
            
        # Generate customer load for each staffing period
        for time in times:
            # +1 to avoid no demand situation (demand = 0)
            poissonParams[str(time)] = np.random.poisson(
                    rates[times.index(time)]) + 1  
        
        # Generate incoming customers (service time follows exponential law)
        for time in times:
            demand = poissonParams[str(time)]
            for i in range(demand):
                Customers.append(Customer(shapeGamma,
                                          scaleGamma,
                                          float(time)
                                          )
                )
                
        # Simulate queuing
        time = times[0]
        count = count_per_interval[str(time)]
        workers = [time for i in range(int(count))]
        
        for i in range(len(Customers)):
            if Customers[i].arrival_time > time:
                time = int(Customers[i].arrival_time)
                count = count_per_interval[str(time)]
                workers = [time for t in range(int(count))] 
            workers = sorted(workers)
            
            if workers[0] >= time + interval_length:
                k = (workers[0] - time)//interval_length
                time += k*interval_length
                try:
                    count = count_per_interval[str(int(time))]
                    workers = [time for t in range(int(count))] 
                except:
                    for j in range(i,len(Customers)):
                        Customers[j].wait = tau + 10000
                    break
            
            start_time = workers[0]
            Customers[i].service_start_time = start_time 
            Customers[i].service_end_time = Customers[i].service_start_time+Customers[i].service_time
            Customers[i].wait = Customers[i].service_start_time - Customers[i].arrival_time
            workers[0] = Customers[i].service_end_time
        
        # Flag customers with exceeding waiting times
        demand_amplitude = deepcopy(poissonParams)
        waiting_times = dict.fromkeys(list(demand_amplitude), 0)
        for customer in Customers:
            if customer.wait > tau:
                waiting_times[str(int(customer.arrival_time))] += 1
        
        # Generate performance summary for each staffing interval
        for time in waiting_times:
            waiting_times[time] /= demand_amplitude[time]
            try:
                waiting_times_summary[time] += waiting_times[time]
            except:
                waiting_times_summary[time] = waiting_times[time]
    
        # Compare performance to threshold for each staffing interval
        # Return lowest under-performing staffing interval if any.
        for time in waiting_times_summary:
            waiting_times_summary[time] /= nbOccurences
        if waiting_times_summary[time] > thres:
            return False, int(int(time)/interval_length)
        
    return True, None 
