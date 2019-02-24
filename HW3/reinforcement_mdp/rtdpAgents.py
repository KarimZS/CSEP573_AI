# rtdpAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu). The RTDP question was added by
# Gagan Bansal (bansalg@cs.washington.edu) and Dan Weld (weld@cs.washington.edu).


import random
import mdp, util, time

from learningAgents import ValueEstimationAgent

class RTDPAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A RTDPAgent takes a Markov decision process
        (see mdp.py) on initialization and runs rtdp 
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, max_iters=100):
        """
          Your value rtdp agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
              mdp.getStartState()

          Other useful functions:
              weighted_choice(choices)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}  # note, we use a normal python dictionary for RTDPAgent.

        # Write rtdp code here
        "*** YOUR CODE HERE ***"
        start_time = time.time()
        totalVals = []
        totalTimes = []
        for i in range(iterations):
            self.RTDPTrialReverse(self.mdp.getStartState())
            totalVals.append(self.values[self.mdp.getStartState()])
            totalTimes.append(time.time()-start_time)
        print(totalVals)
        print(totalTimes)

    def RTDPTrial(self, state):
        while not self.mdp.isTerminal(state):
            action = self.computeActionFromValues(state)
            self.updateValue(state, action)
            state = self.pickNextState(state, action)

    def RTDPTrialReverse(self, state):
        stack = util.Stack()
        while not self.mdp.isTerminal(state):
            action = self.computeActionFromValues(state)
            stack.push((state, action))
            state = self.pickNextState(state, action)

        while not stack.isEmpty():
            state, action = stack.pop()
            self.updateValue(state, action)
    
    def pickNextState(self, state, action):
        """
          Return the next stochastically simulated state.
        """
        "*** YOUR CODE HERE ***"
        t = self.mdp.getTransitionStatesAndProbs(state, action)
        return weighted_choice(t)

    def updateValue(self, state, action):
        """
          Update the value of given state.
        """
        "*** YOUR CODE HERE ***"
        value = self.computeQValueFromValues(state, action)
        self.values[state] = value

    def getHeuristicValue(self, state):
        """
          Return the heuristic value of state.
        """
        "*** YOUR CODE HERE ***"
        # your heuristic function here
        if state is self.mdp.getGoalState():
            return self.mdp.getGoalReward()
        if self.mdp.isTerminal(state):
            return 0
        dist = util.manhattanDistance(state, self.mdp.getGoalState())
        return self.discount**(dist) * self.mdp.getGoalReward()

    def getValue(self, state):
        """
          Return the current stored value of the state.
          If the state has not been seen yet then return it heuristic value.

          Note the difference between this and the similar method in valueIterationAgents
        """
        if state not in self.values:
            self.values[state] = self.getHeuristicValue(state)
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for t in self.mdp.getTransitionStatesAndProbs(state, action):
            q = q + (t[1] * (self.discount * self.getValue(t[0]) + self.mdp.getReward(state, action, t[0])))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        max = None
        best_action = None
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            if max == None or q > max:
                max = q
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


def weighted_choice(choices):
    """
    Return a random element from list of the form: [(choice, weight), ....]
    Credits: http://stackoverflow.com/questions/3679694
    """
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
       if upto + w >= r:
          return c
       upto += w
    assert False, "Shouldn't get here"
