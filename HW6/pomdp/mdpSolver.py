"""
UW, CSEP 573, Win19
"""
from pomdp import POMDP
from offlineSolver import OfflineSolver
import numpy as np


class QMDP(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        super(QMDP, self).__init__(pomdp, precision)
        """
        ****Your code
        Remember this is an offline solver, so compute the policy here
        """
        self.states = pomdp.states
        self.actions = pomdp.actions
        self.values = np.zeros(len(self.states))
        self.qvalues = np.zeros((len(self.states), len(self.actions)))
        self.precision = precision
        self.discount = pomdp.discount
        self.T = pomdp.T.copy()
        self.reward = self.computeReward(pomdp)
        self.runValueIteration()

    def chooseAction(self, cur_belief):
        """
        ***Your code
        """
        maxValue = -999999
        maxAction = None
        for action in range(len(self.actions)):
             q= self.qMDP(action, cur_belief)
             if(q>maxValue):
                 maxValue=q
                 maxAction= action
        return maxAction

    
    def getValue(self, belief):
        """
        ***Your code
        """
        maxValue = -999999
        for action in range(len(self.actions)):
            actionValue = self.qMDP(action, belief)
            if (actionValue > maxValue):
                maxValue = actionValue
        return maxValue
    

    """
    ***Your code
    Add any function, data structure, etc that you want
    """
    def runValueIteration(self):
        while True:
            iterationValues = np.zeros(len(self.states))
            iterationQvalues = np.zeros((len(self.states), len(self.actions)))
            for state in range(len(self.states)):
                maxValue = None
                for action in range(len(self.actions)):
                    tempQ = self.computeQValueFromValues(state, action)
                    iterationQvalues[state,action] = tempQ
                    if maxValue is None or tempQ>=maxValue:
                        maxValue = tempQ
                iterationValues[state] = maxValue

            diff = []
            for i in range(len(self.values)):
                diff.append(abs(self.values[i] - iterationValues[i]))
            maxDiff = max(diff)

            self.values = iterationValues
            self.qvalues = iterationQvalues
            if maxDiff <= self.precision:
                break

    def computeQValueFromValues(self, state, action):
        q = np.dot(self.T[action,state,:],self.values)
        return q*self.discount + self.reward[state,action]

    def computeReward(self, pomdp):
        """for i in range(len(pomdp.actions)):
            for j in range(len(pomdp.states)):
                for k in range(len(pomdp.states)):
                    reward[j,i] = reward[j,i]+(pomdp.R[i,j,k,0]*pomdp.T[i,j,k])
        return reward
        """
        temp = np.multiply(pomdp.R[:,:,:,0],pomdp.T)
        sum = np.sum(temp,2)
        return np.swapaxes(sum,0,1)

    def qMDP(self,action, belief):
        actionQ= self.qvalues[:,action]
        return np.dot(belief,actionQ)

class MinMDP(OfflineSolver):
    
    def __init__(self, pomdp, precision = .001):
        super(MinMDP, self).__init__(pomdp, precision)
        """
        ***Your code 
        Remember this is an offline solver, so compute the policy here
        """
        self.R = pomdp.R
        self.actions = pomdp.actions
        self.discount = pomdp.discount
        self.reward = self.computeReward(pomdp)
    
    def getValue(self, cur_belief):
        """
        ***Your code
        """
        rmin = np.min(self.R)
        maxValue = -999999
        for action in range(len(self.actions)):
            actionValue = np.dot(cur_belief, self.reward[:, action])
            if (actionValue > maxValue):
                maxValue = actionValue
        return maxValue + (self.discount / (1-self.discount))* rmin


    def chooseAction(self, cur_belief):
        """
        ***Your code
        """  
        maxValue = -999999
        maxAction = None
        for action in range(len(self.actions)):
            q=np.dot(cur_belief, self.reward[:,action])
            if(q>maxValue):
                maxValue= q
                maxAction =action
        return maxAction

    """
    ***Your code
    Add any function, data structure, etc that you want
    """
    def computeReward(self, pomdp):
        """reward=np.zeros((len(pomdp.states), len(pomdp.actions)))
        for i in range(len(pomdp.actions)):
            for j in range(len(pomdp.states)):
                for k in range(len(pomdp.states)):
                    reward[j,i] = reward[j,i]+(pomdp.R[i,j,k,0]*pomdp.T[i,j,k])
        return reward
        """
        temp = np.multiply(pomdp.R[:,:,:,0],pomdp.T)
        sum = np.sum(temp,2)
        return np.swapaxes(sum,0,1)