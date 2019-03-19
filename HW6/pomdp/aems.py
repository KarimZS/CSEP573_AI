#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from onlineSolver import OnlineSolver
import numpy as np

class AEMS2(OnlineSolver):
    def __init__(self, pomdp, lb_solver, ub_solver, precision = .001, action_selection_time = .1):
        super(AEMS2, self).__init__(pomdp, precision, action_selection_time)
        self.lb_solver = lb_solver
        self.ub_solver = ub_solver
        self.root = OrNode(pomdp.prior)
        self.root.upperbound = self.ub_solver.getValue(pomdp.prior)
        self.root.lowerbound = self.lb_solver.getValue(pomdp.prior)
        self.root.parent = None
        self.root.depth = 0
        self.root.bestActionIndex = self.ub_solver.chooseAction(pomdp.prior)
        self.root.prob = 1
        self.root.discount = pomdp.discount
        self.root.updateError()
        self.frontier = [self.root]
        self.reward = self.computeReward(pomdp)

        """
        *****Your code
        You can add any attribute you want
        """

    def expandOneNode(self):
        """
        *****Your code
        """
        maxError = -999999
        maxErrorNode = None
        for i in range(len(self.frontier)):
            if self.frontier[i].error > maxError:
                maxError = self.frontier[i].error
                maxErrorNode = self.frontier[i]

        if maxError < self.precision:
            return False
        self.frontier.remove(maxErrorNode)

        for action in range(len(self.pomdp.actions)):
            actionNode = AndNode(action)
            actionNode.parent = maxErrorNode
            actionNode.discount = self.pomdp.discount
            actionNode.reward = self.computeActionReward(action, maxErrorNode.belief)
            actionNode.updateProb()
            maxErrorNode.children.append(actionNode)

            for observation in range(len(self.pomdp.observations)):
                newBeliefNoObeservation = np.matmul(maxErrorNode.belief, self.pomdp.T[action, :, :])
                newBelief = newBeliefNoObeservation * self.pomdp.O[action, :, observation]
                total = np.sum(newBelief)
                if (total != 0):
                    newBelief = newBelief / total

                newBeliefNode = OrNode(newBelief)
                newBeliefNode.discount = self.pomdp.discount
                newBeliefNode.observation = self.computeObservation(observation, maxErrorNode.belief, action)
                newBeliefNode.observationIndex = observation

                newBeliefNode.lowerbound = self.lb_solver.getValue(newBelief)
                newBeliefNode.upperbound = self.ub_solver.getValue(newBelief)

                newBeliefNode.bestActionIndex = self.ub_solver.chooseAction(newBelief)
                newBeliefNode.updateProb()

                newBeliefNode.updateError()
                newBeliefNode.parent=actionNode
                actionNode.children.append(newBeliefNode)
                self.frontier.append(newBeliefNode)

            actionNode.updateBounds()
        self.backprop(maxErrorNode)

        return True

    
    def chooseAction(self):
        """
        *****Your code
        """
        maxAction = 0 #default if node isnt expanded
        maxUpper = -999999
        for actionNode in self.root.children:
            if actionNode.upperbound>maxUpper:
                maxUpper =  actionNode.upperbound
                maxAction = actionNode.action
        return maxAction
        
    def updateRoot(self, action, observation):
        """
        ***Your code 
        """
        newRootNode = None
        for actionNode in self.root.children:
            if actionNode.action != action:
                continue

            for beliefNode in actionNode.children:
                if beliefNode.observationIndex != observation:
                    continue
                newRootNode = beliefNode
                break
            break

        if newRootNode == None:#the action we took was on an unexpanded belief
            belief = np.matmul(self.root.Belief , self.pomdp.T[action, :, :])
            belief = belief * self.pomdp.O[action, :, observation]
            total = np.sum(belief)
            if(total!= 0):
                belief = belief / total
            beliefNode =OrNode(belief)
            beliefNode.discount = self.pomdp.discount
            beliefNode.observation = self.computeObservation(observation, self.root.belief, action)
            beliefNode.observationIndex= observation
            beliefNode.lowerbound= self.lb_solver.getValue(belief)
            beliefNode.upperbound = self.ub_solver.getValue(belief)
            beliefNode.bestActionIndex = self.ub_solver.chooseAction(belief)
            newRootNode= beliefNode
        newRootNode.parent = None
        beliefNode.depth = 0
        beliefNode.prob = 1
        self.updateChildProbAndDepth(newRootNode)
        newRootNode.updateError()
        self.root = newRootNode
        self.frontier = []
        self.recreateFrontier(self.root)

    def updateChildProbAndDepth(self,root):
        root.updateProb()
        for child in root.children:
            self.updateChildProbAndDepth(child)


    def recreateFrontier(self, root):
        if len(root.children) == 0:
            self.frontier.append(root)
        else:
            for child in root.children[root.bestActionIndex].children:
                self.recreateFrontier(child)

    def backprop(self,node):
        current = node
        while current != None:
            updated = current.updateBounds()
            if updated==False:
                return
            current = current.parent

    def computeReward(self,pomdp):
        temp = np.multiply(pomdp.R[:,:,:,0],pomdp.T)
        sum = np.sum(temp,2)
        return np.swapaxes(sum,0,1)

    def computeActionReward(self, action, belief):
        return np.dot(belief, self.reward[:,action])

    def computeObservation(self, observation, belief, action):
        current_belief = np.matmul(belief, self.pomdp.T[action, :, :])
        current_belief = np.dot(current_belief , self.pomdp.O[action, :, observation])
        return current_belief

"""
****Your code 
add any data structure, code, etc you want 
We recommend to have a super class of Node and two subclasses of AndNode and OrNode 
"""

class Node():
    def __init__(self):
        self.parent = None
        self.upperbound= None
        self.lowerbound = None
        self.children = []
        self.discount = None
        self.depth = None
        self.prob = None

class AndNode(Node):
    def __init__(self,action):
        super(AndNode,self).__init__()
        self.action = action
        self.reward = None

    def updateBounds(self):
        currentLower = self.lowerbound
        currentUpper = self.upperbound
        self.lowerbound = 0
        self.upperbound = 0
        for belief in self.children:
            self.lowerbound += belief.lowerbound * belief.observation
            self.upperbound += belief.upperbound * belief.observation

        self.upperbound *= self.discount
        self.lowerbound *= self.discount
        self.upperbound += self.reward
        self.lowerbound += self.reward
        if currentLower==self.lowerbound and currentUpper==self.upperbound:
            return True
        else:
            return False
    def updateProb(self):
        if self.parent.bestActionIndex is self.action:
            self.prob = self.parent.prob
        else:
            self.prob=0
        self.depth = self.parent.depth

class OrNode(Node):
    def __init__(self,belief):
        super(OrNode,self).__init__()
        self.belief = belief
        self.error = None
        self.observation = None
        self.observationIndex = None
        self.bestActionIndex = None

    def updateBounds(self):
        currentLower = self.lowerbound
        currentUpper = self.upperbound
        self.lowerbound = -999999
        self.upperbound = -999999
        for action in self.children:
            self.lowerbound = max(self.lowerbound, action.lowerbound)
            self.upperbound = max(self.upperbound, action.upperbound)
        if currentLower==self.lowerbound and currentUpper==self.upperbound:
            return True
        else:
            return False
    def updateProb(self):
        if self.parent is not None:
            self.prob = self.parent.prob*self.observation
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
            self.prob = 1
    def updateError(self):
        self.error = (self.discount**self.depth) * self.prob * (self.upperbound - self.lowerbound)
