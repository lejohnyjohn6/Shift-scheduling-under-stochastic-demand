#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 2018
Author: John Pougué Biyong
Contact: john.pougue-biyong@maths.ox.ac.uk

This script aims at optimizing shift schedulling under stochastic demand
by using a simulation-based branch-and-bound technique introduced by Defraeye et al. (2015)
Please refer to project report for further details.

"""

import numpy as np
from copy import deepcopy
import pulp as plp
import time as timeLib
import simulation as simul

class Node:
    
    def __init__(self, staffingCosts, parent, depth, capacities, 
                 shift_vector=None, lp_shift_cost=None, ip_shift_cost=None):
        
        self.parent = parent
        self.depth = depth
        self.capacities = capacities
        self.staffing_cost = self._compute_staffing_cost(staffingCosts)
        self.shift_vector = shift_vector
        self.lp_shift_cost = lp_shift_cost
        self.ip_shift_cost = ip_shift_cost
    
    def print_details(self):
        """ Prints the details of the node. """
        
        print("parent:", self.parent)
        print("depth:", self.depth)
        print("capacities:", self.capacities)
        print("staffing_cost:", self.staffing_cost)
        print("shift_vector:", self.shift_vector)
        print("lp_shift_cost:", self.lp_shift_cost)
        print("ip_shift_cost:", self.ip_shift_cost)
    
    def increase_capacity(self, depth, count):
        """ Increases capacity at node in depth 'depth'. """
        
        new_capacities = deepcopy(self.capacities)
        new_capacities[depth] += count
        return new_capacities
    
    def _compute_staffing_cost(self, staffingCosts):
        
        return sum(staffingCosts*self.capacities)
        
    def optimize_lp(self, shiftCosts, shiftMatrix, nbIntervals):
        """ Solves LP relaxation for set covering problem. """
        
        # Define problem
        staffing_level = self.capacities
        scp = plp.LpProblem("Set covering problem (LP)")
        
        # Define variables
        d = {}
        for x in range(len(shiftCosts)):
            d["x{}".format(x+1)] = 0
        variables = plp.LpVariable.dicts("shift", list(d), 0)
        
        # Objective function
        scp += plp.lpSum([shiftCosts[int(i[-1])-1]*variables[i] for i in variables])
        
        # Constraints
        for i in range(nbIntervals):
            v = shiftMatrix[:,i]
            scp += plp.lpSum(
                    [v[int(j[-1])-1]*variables[j] for j in list(d)]
                    ) >= staffing_level[i]
        
        # Solution
        scp.solve()
        
        return plp.value(scp.objective), [v.varValue for v in scp.variables()]
    
    def optimize_ip(self, shiftCosts, shiftMatrix, nbIntervals):
        """ Solves set covering problem (integer constraint). """
        
        # Define problem
        staffing_level = self.capacities
        scp = plp.LpProblem("Set covering problem (IP)")
        
        # Define variables
        d = {}
        for x in range(len(shiftCosts)):
            d["x{}".format(x+1)] = 0
        variables = plp.LpVariable.dicts("shift", list(d), 0, cat='Integer')
        
        # Objective function
        scp += plp.lpSum(
                [shiftCosts[int(i[-1])-1]*variables[i] for i in variables]
                )
        
        # Constraints
        for i in range(nbIntervals):
            v = shiftMatrix[:,i]
            scp += plp.lpSum(
                    [v[int(j[-1])-1]*variables[j] for j in list(d)]
                    ) >= staffing_level[i]

        # Solution
        scp.solve()
        
        return plp.value(scp.objective), [v.varValue for v in scp.variables()]
        
""" Branch-and-bound methods """

def Backtrack(s, d, currentNode, sUB, sLB):
    """ Returns to previous level and proceeds with next node. """

    if d != -1:
        while (s[d] == sUB[d] and d > -1):
            s[d] = deepcopy(sLB[d])
            d -= 1
            currentNode = currentNode.parent
        if d != -1:
            try:
                currentNode = Node(staffingCosts,
                               parent=currentNode.parent,
                               depth=d,
                               capacities=currentNode.increase_capacity(d,1)
                               )   
            except:
                currentNode.print_details()
            s[d] = deepcopy(s[d]) + 1
      
    return s, d, currentNode

def Branch(i, s, currentNode, sUB, sLB):
    """ Branches to child node on level i. """
    
    dNew = i
    currentNode = Node(staffingCosts,
                       parent=currentNode,
                       depth=dNew,
                       capacities=s
                       ) 
    if s[dNew] < sUB[dNew]:
        s[dNew] += 1
        currentNode = Node(staffingCosts,
                           parent=currentNode.parent,
                           depth=dNew,
                           capacities=s
                           )
    else:
        s, dNew, currentNode = Backtrack(s, dNew, currentNode, sUB, sLB)
    
    return s, dNew, currentNode

""" Branch-and-bound method """
    
def main(nbIntervals, interval_length, sInit, sLB, sUB, wInit, shiftMatrix, 
         shiftCosts, staffingCosts, rates, shapeGamma, scaleGamma, times,
         tau, thres, x):   
    """ Optimizes shift schedulling by using a branch-and-bound technique 
    coupled with simulation. """
    
    # Initialization
    w_star = wInit
    s_tilde_w_star = sInit 
    cost_w_star = sum(shiftCosts*wInit)
    
    d = -1
    infeasible_vectors, infeasible_stop_indexes = [], []
    
    # Set tree root
    root = Node(staffingCosts, parent='root has no parent', depth=d, capacities=sLB)
    
    # Set starting node
    first = sUB-sLB
    for i in range(len(first)):
        if first[i] > 0:
            currentNode = Node(staffingCosts,
                               parent=root,
                               depth=i,
                               capacities=root.increase_capacity(i,0)
                          ) 
            break
            
    s = deepcopy(currentNode.capacities)
    currentNode.print_details()
     
    # Browse tree
    d+=1
    while d > -1:
        print('depth: {}'.format(d))
        
        # Challenge staffing cost
        if currentNode.staffing_cost >= cost_w_star:
            # Fathom [Cs]
            print('# Fathom [Cs]')
            s, d, currentNode = Backtrack(s, d, currentNode, sUB, sLB)
            
        else:
            # Solve set covering problem (LP relaxation)
            currentNode.lp_shift_cost = currentNode.optimize_lp(
                    shiftCosts, shiftMatrix, nbIntervals
                    )[0]
            
            # Challenge (LP-optimized) shift cost
            if currentNode.lp_shift_cost >= cost_w_star:
                # Fathom [CwLP]
                print('# Fathom [CwLP]')
                s[d] = deepcopy(sUB[d])
                s, d, currentNode = Backtrack(s, d, currentNode, sUB, sLB)
                
            else:
                # Solve set covering problem (IP)
                currentNode.shift_vector = currentNode.optimize_ip(
                    shiftCosts, shiftMatrix, nbIntervals
                    )[1]
                currentNode.ip_shift_cost = currentNode.optimize_ip(
                    shiftCosts, shiftMatrix, nbIntervals
                    )[0]
                
                # Challenge (IP-optimized) shift cost
                if currentNode.ip_shift_cost >= cost_w_star:
                    # Fathom [Cw]
                    print('# Fathom [Cw]')
                    s[d] = deepcopy(sUB[d])
                    s, d, currentNode = Backtrack(s, d, currentNode, sUB, sLB)
                    
                else:
                    # Check if w has been simulated before
                    if currentNode.shift_vector in infeasible_vectors:
                        print('simulated before')
                        ie = infeasible_stop_indexes[
                                infeasible_vectors.index(
                                        currentNode.shift_vector
                                        )
                                ]
                        if d < ie:
                            print('# Branch [ie]')
                            # Proceed to child node at depth ie
                            s, d, currentNode = Branch(ie, s, currentNode, sUB, sLB)
                        else:
                            print('# Fathom [ie]')
                            # Fathom [ie]
                            while d > ie:
                                s[d] = sLB[d]
                                if currentNode.depth > 0:
                                    currentNode = Node(staffingCosts,
                                                       parent=root,
                                                       depth=currentNode.depth-1,
                                                       capacities=s)
                                d -= 1
                            s, d, currentNode = Backtrack(s, d, currentNode, sUB, sLB)
                            
                    else:
                        # Check if w satisfies service level requirements
                        print('not simulated before')
                        test, stop_index = simul.Simulation(currentNode, rates,
                                                    shapeGamma, scaleGamma,
                                                    times, shiftMatrix, tau, thres,
                                                    x, interval_length) 
                        
                        # Update estimated optimum if w satisties service level req.
                        if test == True:
                            print('NEW STAR!')
                            w_star = currentNode.shift_vector
                            s_tilde_w_star = np.array([0 for i in range(nbIntervals)])
                            for i in range(s_tilde_w_star.shape[0]):
                                s_tilde_w_star[i] = sum(shiftMatrix[:,i]*w_star)
                            cost_w_star = currentNode.ip_shift_cost
                            s[d] = deepcopy(sUB[d])
                            s, d, currentNode = Backtrack(s, d, currentNode, sUB, sLB)
                            
                        elif test == False: 
                            print('not new star')
                            ie = stop_index
                            infeasible_vectors += [currentNode.shift_vector]
                            infeasible_stop_indexes += [ie]
                            if d < ie:
                                print('# Branch [ie]')
                                # Proceed to child node at depth ie
                                s, d, currentNode = Branch(ie, s, currentNode, sUB, sLB)
                                
                            else:
                                print('# Fathom [ie]')
                                # Fathom [ie]
                                while d > ie:
                                    s[d] = sLB[d]
                                    if 1 > 0:
                                    #if currentNode.parent == root or currentNode.parent == 'root has no parent':
                                        if currentNode.depth > 0:
                                            currentNode = Node(staffingCosts,
                                                               parent=root,
                                                               depth=currentNode.depth-1,
                                                               capacities=s)
                                            
                                    else:
                                        currentNode = currentNode.parent
                                    d -= 1
                                s, d, currentNode = Backtrack(s, d, currentNode, sUB, sLB)
   
    return w_star, s_tilde_w_star, cost_w_star

if __name__ == '__main__':
    # Working hours
    working_hours = {'start': '09:00', 'end': '22:00'}
    start = working_hours['start']
    end = working_hours['end']
    int_length = 20
    delta = 60/int_length
    
    # Staffing intervals
    times = [int_length*i for i in range(int((int(end[:2])-int(start[:2]))*delta))]
    
    # Demand rates
    lambda0 = 10
    R = 0.12
    c1 = 2*np.pi/(1*420)
    c2 = (np.pi/2)-600*c1/2
    rates = []
    for time in times:
        rates += [lambda0*(1 + R*np.cos(c1*time+c2))]
        
    # Parameters
    shapeGamma = 1
    scaleGamma = 16
    tau = 9
    thres = 0.01
    simulOccurences = 10
    
    # Shift types
    shifts = {'shift1': {'start': '09:00', 'end': '11:00'},
              'shift2': {'start': '11:00', 'end': '12:00'},
              'shift3': {'start': '11:20', 'end': '12:20'},
              'shift4': {'start': '12:20', 'end': '14:40'},
              'shift5': {'start': '14:20', 'end': '17:00'},
              'shift6': {'start': '17:00', 'end': '20:20'},
              'shift7': {'start': '18:20', 'end': '19:20'},
              'shift8': {'start': '20:00', 'end': '22:00'}
             }
    
    # Costs
    staffingCosts = np.array([1/delta for item in times])
    shiftCosts = np.array([120, 60, 60, 140, 160, 200, 60, 120])
    shiftCosts = shiftCosts/60
    nbIntervals = len(staffingCosts)
    
    # Shift types
    shiftMatrix = np.zeros([len(shiftCosts), len(staffingCosts)])
    for i in range(len(shiftCosts)):
        startTime = shifts['shift{}'.format(i+1)]['start']
        endTime = shifts['shift{}'.format(i+1)]['end']
        startTime = int(startTime[:2])*60 + int(startTime[-2:]) - 9*60
        endTime = int(endTime[:2])*60 + int(endTime[-2:]) - 9*60
        for j in range(len(staffingCosts)):
            if startTime <= times[j] < endTime:
                shiftMatrix[i][j] = 1
    
    # Vector input sLB
    sLB = np.ones(len(times))
    
    # Vector input sUB
    sUB = np.zeros(len(times))
    for time in times:
        lamb = rates[times.index(time)]
        value = 0
        dummy = np.exp(-lamb)
        count = 0
        while value < 1 - thres:
            value += dummy
            dummy *= lamb/(count+1)
            count += 1
        sUB[times.index(time)] = count
    
    # Vector input sInit
    sInit = deepcopy(sUB)
    nodeInit = Node(staffingCosts,
                    parent='x',
                    depth=0,
                    capacities=sInit
               )
    wInit = nodeInit.optimize_ip(
                    shiftCosts, shiftMatrix, nbIntervals
                    )[1] 
    
    # Problem solver
    optimizationOutput = main(nbIntervals, int_length, sInit, sLB, sUB, wInit,
                              shiftMatrix, shiftCosts, staffingCosts, rates, 
                              shapeGamma, scaleGamma, times, tau, thres, 
                              simulOccurences)
    print(optimizationOutput)
