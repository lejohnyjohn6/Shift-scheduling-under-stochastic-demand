#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 2018
Author: John Pougué Biyong
Contact: jpougue@gmail.com

This module gathers support functions for the branh-and-bound method.
Please refer to project report for further details.

"""

from copy import deepcopy
import numpy as np

def generate_service_time(shape, scale=1):
    """ Generates 1 realisation of a exponential-distributed random variable. 
    This counts as the duration for a customer service. """
    return  np.random.gamma(shape, scale, 1)

class Customer:
    """ Simply defines methods for the customers. """
    
    def __init__(self, shape, scale, arrival_time, service_start_time=None, wait=None):
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
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
        
def simulate(node, rates, shape_gamma, scale_gamma, times, shift_matrix, 
             tau, thres, nb_occurences, interval_length): 
    """ Simulates operations over the planning horizon with specified shift vector (repeats 'nb_occurences' times). 
    Returns True if the quality service constraints is satisfied. Otherwise, returns False and the index of the 
    first staffing period at which the constraint has been violated. """
    waiting_times_summary = {}
    
    # Simulate 'nb_occurences' times
    for occurence in range(nb_occurences):
        poisson_params = {}
        customers = []
        count_per_interval = {}
        
        # Set workforce for each staffing period
        for time in times:
            index = int(time/interval_length)
            count_per_interval[str(time)] = sum(shift_matrix[:, index]*node.shift_vector)
            
        # Generate customer load for each staffing period
        for time in times:
            # +1 to avoid no demand situation (demand = 0)
            poisson_params[str(time)] = np.random.poisson(rates[times.index(time)]) + 1  
        
        # Generate incoming customers (service time follows exponential law)
        for time in times:
            demand = poisson_params[str(time)]
            for i in range(demand):
                customers.append(Customer(shape_gamma, scale_gamma, float(time)))
                
        # Simulate queuing
        time = times[0]
        count = count_per_interval[str(time)]
        workers = [time for i in range(int(count))]
        nb_customers = len(customers)
        for i in range(nb_customers):
            if customers[i].arrival_time > time:
                time = int(customers[i].arrival_time)
                count = count_per_interval[str(time)]
                workers = [time for t in range(int(count))] 
            workers = sorted(workers)
            
            if workers[0] >= time + interval_length:
                k = (workers[0] - time)//interval_length
                time += k*interval_length
                try:
                    count = count_per_interval[str(int(time))]
                    workers = [time for t in range(int(count))] 
                except KeyError:
                    for j in range(i, len(customers)):
                        customers[j].wait = tau + 10000
                    break
            
            start_time = workers[0]
            customers[i].service_start_time = start_time 
            customers[i].service_end_time = customers[i].service_start_time+customers[i].service_time
            customers[i].wait = customers[i].service_start_time - customers[i].arrival_time
            workers[0] = customers[i].service_end_time
        
        # Flag customers with exceeding waiting times
        demand_amplitude = deepcopy(poisson_params)
        waiting_times = dict.fromkeys(list(demand_amplitude), 0)
        for customer in customers:
            if customer.wait > tau:
                waiting_times[str(int(customer.arrival_time))] += 1
        
        # Generate performance summary for each staffing interval
        for time in waiting_times:
            waiting_times[time] /= demand_amplitude[time]
            try:
                waiting_times_summary[time] += waiting_times[time]
            except KeyError:
                waiting_times_summary[time] = waiting_times[time]
    
        # Compare performance to threshold for each staffing interval.
        # Return lowest under-performing staffing interval if any.
        for time in waiting_times_summary:
            waiting_times_summary[time] /= nb_occurences
        if waiting_times_summary[time] > thres:
            return False, int(int(time)/interval_length)
        
    return True, None 
