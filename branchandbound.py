#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 2018
Author: John Pougué Biyong
Contact: jpougue@gmail.com

This script aims at optimizing shift schedulling under stochastic demand
by using a simulation-based branch-and-bound technique introduced by Defraeye et al. (2015)
Please refer to project report for further details.

"""

from copy import deepcopy
import numpy as np
import pulp as plp
import simulation as simul

class Node:
    """ Simply defines methods for the tree nodes. """
    
    def __init__(self, staffing_costs, parent, depth, capacities,
                 shift_vector=None, lp_shift_cost=None, ip_shift_cost=None): 
        self.parent = parent
        self.depth = depth
        self.capacities = capacities
        self.staffing_cost = self._compute_staffing_cost(staffing_costs)
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
    
    def _compute_staffing_cost(self, staffing_costs):
        return sum(staffing_costs*self.capacities)
        
    def optimize_lp(self, shift_costs, shift_matrix, nb_intervals):
        """ Solves LP relaxation for set covering problem. """
        # Define problem
        staffing_level = self.capacities
        scp = plp.LpProblem("Set covering problem (LP)")
        
        # Define variables
        dico = {}
        for cost_index in range(len(shift_costs)):
            dico["x{}".format(cost_index+1)] = 0
        variables = plp.LpVariable.dicts("shift", list(dico), 0)
        
        # Objective function
        scp += plp.lpSum([shift_costs[int(var[-1])-1]*variables[var] for var in variables])
        
        # Constraints
        for interval in range(nb_intervals):
            column = shift_matrix[:, interval]
            scp += plp.lpSum([column[int(var[-1])-1]*variables[var] for var in list(dico)]) >= staffing_level[interval]
        
        # Solution
        scp.solve()
        
        return plp.value(scp.objective), [v.varValue for v in scp.variables()]
    
    def optimize_ip(self, shift_costs, shift_matrix, nb_intervals):
        """ Solves set covering problem (integer constraint). """
        # Define problem
        staffing_level = self.capacities
        scp = plp.LpProblem("Set covering problem (IP)")
        
        # Define variables
        dico = {}
        for cost_index in range(len(shift_costs)):
            dico["x{}".format(cost_index+1)] = 0
        variables = plp.LpVariable.dicts("shift", list(dico), 0, cat='Integer')
        
        # Objective function
        scp += plp.lpSum([shift_costs[int(var[-1])-1]*variables[var] for var in variables])
        
        # Constraints
        for interval in range(nb_intervals):
            column = shift_matrix[:, interval]
            scp += plp.lpSum([column[int(var[-1])-1]*variables[var] for var in list(dico)]) >= staffing_level[interval]

        # Solution
        scp.solve()
        
        return plp.value(scp.objective), [v.varValue for v in scp.variables()]

def backtrack(current_staffing_vector, current_depth, current_node, s_ub, s_lb):
    """ Returns to previous level and proceeds with next node. """
    if current_depth != -1:
        while (current_staffing_vector[current_depth] == s_ub[current_depth] and current_depth > -1):
            current_staffing_vector[current_depth] = deepcopy(s_lb[current_depth])
            current_depth -= 1
            current_node = current_node.parent
        if current_depth != -1:
            current_node = Node(STAFFING_COSTS,
                                parent=current_node.parent,
                                depth=current_depth,
                                capacities=current_node.increase_capacity(current_depth, 1)
                                )  
            current_staffing_vector[current_depth] = deepcopy(current_staffing_vector[current_depth]) + 1
                  
    return current_staffing_vector, current_depth, current_node

def branch(i, current_staffing_vector, current_node, s_ub, s_lb):
    """ branches to child node on level i. """
    d_new = i
    current_node = Node(STAFFING_COSTS,
                        parent=current_node,
                        depth=d_new,
                        capacities=current_staffing_vector) 
    if current_staffing_vector[d_new] < s_ub[d_new]:
        current_staffing_vector[d_new] += 1
        current_node = Node(STAFFING_COSTS,
                            parent=current_node.parent,
                            depth=d_new,
                            capacities=current_staffing_vector)
    else:
        current_staffing_vector, d_new, current_node = backtrack(current_staffing_vector,
                                                                 d_new,
                                                                 current_node,
                                                                 s_ub,
                                                                 s_lb)
    
    return current_staffing_vector, d_new, current_node
    
def main(nb_intervals, interval_length, s_init, s_lb, s_ub, w_init, shift_matrix, shift_costs, staffing_costs, 
         rates, shape_gamma, scale_gamma, times, tau, thres, simulation_occurences):   
    """ Optimizes shift schedulling by using a branch-and-bound technique 
    coupled with simulation. """
    # Initialization
    w_star = w_init
    s_tilde_w_star = s_init 
    cost_w_star = sum(shift_costs*w_init)
    current_depth = -1
    infeasible_vectors, infeasible_stop_indexes = [], []
    # Set tree root
    root = Node(staffing_costs, parent='root has no parent', depth=current_depth, capacities=s_lb)
    # Set starting node
    first = s_ub-s_lb
    current_node = deepcopy(root)
    for index in range(len(first)):
        if first[index] > 0:
            current_node = Node(staffing_costs,
                                parent=root,
                                depth=index,
                                capacities=root.increase_capacity(index, 0)
                                ) 
            break       
    current_staffing_vector = deepcopy(current_node.capacities)
    current_node.print_details() 
    # Browse tree
    current_depth += 1
    while current_depth > -1:
        print('depth: {}'.format(current_depth))
        # Challenge staffing cost
        if current_node.staffing_cost >= cost_w_star:
            # fathom [Cs]
            print('# fathom [Cs]')
            current_staffing_vector, current_depth, current_node = backtrack(current_staffing_vector,
                                                                             current_depth,
                                                                             current_node,
                                                                             s_ub,
                                                                             s_lb)    
        else:
            # Solve set covering problem (LP relaxation)
            current_node.lp_shift_cost = current_node.optimize_lp(shift_costs, shift_matrix, nb_intervals)[0]
            # Challenge (LP-optimized) shift cost
            if current_node.lp_shift_cost >= cost_w_star:
                # fathom [CwLP]
                print('# fathom [CwLP]')
                current_staffing_vector[current_depth] = deepcopy(s_ub[current_depth])
                current_staffing_vector, current_depth, current_node = backtrack(current_staffing_vector, 
                                                                                 current_depth,
                                                                                 current_node,
                                                                                 s_ub,
                                                                                 s_lb)    
            else:
                # Solve set covering problem (IP)
                current_node.shift_vector = current_node.optimize_ip(shift_costs, shift_matrix, nb_intervals)[1]
                current_node.ip_shift_cost = current_node.optimize_ip(shift_costs, shift_matrix, nb_intervals)[0]
                # Challenge (IP-optimized) shift cost
                if current_node.ip_shift_cost >= cost_w_star:
                    # fathom [Cw]
                    print('# fathom [Cw]')
                    current_staffing_vector[current_depth] = deepcopy(s_ub[current_depth])
                    current_staffing_vector, current_depth, current_node = backtrack(current_staffing_vector,
                                                                                     current_depth,
                                                                                     current_node,
                                                                                     s_ub,
                                                                                     s_lb)   
                else:
                    # Check if w has been simulated before
                    if current_node.shift_vector in infeasible_vectors:
                        print('simulated before')
                        violation_index = infeasible_stop_indexes[infeasible_vectors.index(current_node.shift_vector)]
                        if current_depth < violation_index:
                            print('# branch [ie]')
                            # Proceed to child node at depth "violation_index"
                            current_staffing_vector, current_depth, current_node = branch(violation_index,
                                                                                          current_staffing_vector,
                                                                                          current_node,
                                                                                          s_ub,
                                                                                          s_lb)
                        else:
                            print('# fathom [ie]')
                            # fathom [ie]
                            while current_depth > violation_index:
                                current_staffing_vector[current_depth] = s_lb[current_depth]
                                if current_node.depth > 0:
                                    current_node = Node(staffing_costs,
                                                        parent=root,
                                                        depth=current_node.depth-1,
                                                        capacities=current_staffing_vector)
                                current_depth -= 1
                            current_staffing_vector, current_depth, current_node = backtrack(current_staffing_vector,
                                                                                             current_depth,
                                                                                             current_node,
                                                                                             s_ub,
                                                                                             s_lb)      
                    else:
                        # Check if w satisfies service level requirements
                        print('not simulated before')
                        test, violation_index = simul.simulate(current_node, rates, shape_gamma, scale_gamma, times, 
                                                               shift_matrix, tau, thres, simulation_occurences,
                                                               interval_length) 
                        # Update estimated optimum if w satisties service level req.
                        if test:
                            print('NEW STAR!')
                            w_star = current_node.shift_vector
                            s_tilde_w_star = np.array([0 for index in range(nb_intervals)])
                            for index in range(s_tilde_w_star.shape[0]):
                                s_tilde_w_star[index] = sum(shift_matrix[:, index]*w_star)
                            cost_w_star = current_node.ip_shift_cost
                            current_staffing_vector[current_depth] = deepcopy(s_ub[current_depth])
                            current_staffing_vector, current_depth, current_node = backtrack(current_staffing_vector,
                                                                                             current_depth,
                                                                                             current_node,
                                                                                             s_ub,
                                                                                             s_lb)    
                        elif not test: 
                            print('not new star')
                            infeasible_vectors += [current_node.shift_vector]
                            infeasible_stop_indexes += [violation_index]
                            if current_depth < violation_index:
                                print('# branch [ie]')
                                # Proceed to child node at depth "violation_index"
                                current_staffing_vector, current_depth, current_node = branch(violation_index,
                                                                                              current_staffing_vector,
                                                                                              current_node,
                                                                                              s_ub,
                                                                                              s_lb)  
                            else:
                                print('# fathom [ie]')
                                # fathom [ie]
                                while current_depth > violation_index:
                                    current_staffing_vector[current_depth] = s_lb[current_depth]
                                    if current_node.parent == root or current_node.parent == 'root has no parent':
                                        if current_node.depth > 0:
                                            current_node = Node(staffing_costs, 
                                                                parent=root,
                                                                depth=current_node.depth-1,
                                                                capacities=current_staffing_vector)      
                                    else:
                                        current_node = current_node.parent
                                    current_depth -= 1
                                current_staffing_vector, current_depth, current_node = backtrack(current_staffing_vector
                                                                                                 , current_depth
                                                                                                 , current_node
                                                                                                 , s_ub
                                                                                                 , s_lb)
   
    return w_star, s_tilde_w_star, cost_w_star

if __name__ == '__main__':
    # Working hours
    WORKING_HOURS = {'start': '09:00', 'end': '22:00'}
    START = WORKING_HOURS['start']
    END = WORKING_HOURS['end']
    INTERVAL_LENGTH = 20
    DELTA = 60/INTERVAL_LENGTH
    
    # Staffing intervals
    TIMES = [INTERVAL_LENGTH*number for number in range(int((int(END[:2])-int(START[:2]))*DELTA))]
    
    # Demand rates
    LAMBDA = 10
    R = 0.24
    C1 = 2*np.pi/(1*420)
    C2 = (np.pi/2)-600*C1/2
    RATES = []
    for time in TIMES:
        RATES += [LAMBDA*(1 + R*np.cos(C1*time+C2))]
        
    # Parameters
    SHAPE_GAMMA = 1
    SCALE_GAMMA = 16
    TAU = 9
    THRES = 0.01
    SIMULATION_OCCURENCES = 1
    
    # Shift types
    SHIFTS = {'shift1': {'start': '09:00', 'end': '11:00'},
              'shift2': {'start': '11:00', 'end': '12:00'},
              'shift3': {'start': '11:20', 'end': '12:20'},
              'shift4': {'start': '12:20', 'end': '14:40'},
              'shift5': {'start': '14:20', 'end': '17:00'},
              'shift6': {'start': '17:00', 'end': '20:20'},
              'shift7': {'start': '18:20', 'end': '19:20'},
              'shift8': {'start': '20:00', 'end': '22:00'}
             }
    
    # Costs
    STAFFING_COSTS = np.array([1/DELTA for time in TIMES])
    SHIFT_COSTS = np.array([120, 60, 60, 140, 160, 200, 60, 120])
    SHIFT_COSTS = SHIFT_COSTS/60
    NB_INTERVALS = len(STAFFING_COSTS)
    
    # Shift types
    SHIFT_MATRIX = np.zeros([len(SHIFT_COSTS), len(STAFFING_COSTS)])
    for shift_cost_index in range(len(SHIFT_COSTS)):
        START_TIME = SHIFTS['shift{}'.format(shift_cost_index+1)]['start']
        END_TIME = SHIFTS['shift{}'.format(shift_cost_index+1)]['end']
        START_TIME = int(START_TIME[:2])*60 + int(START_TIME[-2:]) - 9*60
        END_TIME = int(END_TIME[:2])*60 + int(END_TIME[-2:]) - 9*60
        for staff_cost_index in range(len(STAFFING_COSTS)):
            if START_TIME <= TIMES[staff_cost_index] < END_TIME:
                SHIFT_MATRIX[shift_cost_index][staff_cost_index] = 1
    
    # Vector input s_lb
    S_LB = np.ones(len(TIMES))
    
    # Vector input s_ub
    S_UB = np.zeros(len(TIMES))
    for time in TIMES:
        LAMB = RATES[TIMES.index(time)]
        VALUE = 0
        DUMMY = np.exp(-LAMB)
        COUNT = 0
        while VALUE < 1 - THRES:
            VALUE += DUMMY
            DUMMY *= LAMB/(COUNT+1)
            COUNT += 1
        S_UB[TIMES.index(time)] = COUNT

    # Vector input s_init
    S_INIT = deepcopy(S_UB)
    NODE_INIT = Node(STAFFING_COSTS,
                     parent='x',
                     depth=0,
                     capacities=S_INIT)
    W_INIT = NODE_INIT.optimize_ip(SHIFT_COSTS, SHIFT_MATRIX, NB_INTERVALS)[1] 
    
    # Problem solver
    BEST_SCHEDULLING = main(NB_INTERVALS, INTERVAL_LENGTH, S_INIT, S_LB, S_UB, W_INIT, SHIFT_MATRIX, SHIFT_COSTS,
                            STAFFING_COSTS, RATES, SHAPE_GAMMA, SCALE_GAMMA, TIMES, TAU, THRES, SIMULATION_OCCURENCES)
    print(BEST_SCHEDULLING)
    
