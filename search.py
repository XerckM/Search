"""
Group Members:

    - Xerckiem Mercado

"""


# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()  # Initialize DFS stack
    stack.push((problem.getStartState(), []))  # Push start state
    visited = set()  # Track visited states

    while not stack.isEmpty():  # While states remain
        state, actions = stack.pop()  # Pop state

        if state in visited:  # Skip visited states
            continue

        visited.add(state)  # Mark state as visited

        if problem.isGoalState(state):  # Check for goal
            return actions

        for nextState, action, _ in problem.getSuccessors(state):  # Iterate successors
            if nextState not in visited:  # Push unvisited successors
                stack.push((nextState, actions + [action]))

    return []  # Return empty list if no solution

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    queue = util.Queue()  # Initialize BFS queue
    queue.push((problem.getStartState(), []))  # Push start state
    visited = set()  # Track visited states

    while not queue.isEmpty():  # While states remain
        state, actions = queue.pop()  # Dequeue state

        if state in visited:  # Skip visited states
            continue

        visited.add(state)  # Mark state as visited

        if problem.isGoalState(state):  # Check for goal
            return actions

        for nextState, action, _ in problem.getSuccessors(state):  # Iterate successors
            if nextState not in visited:  # Enqueue unvisited successors
                queue.push((nextState, actions + [action]))

    return []  # Return empty list if no solution

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    # Utilize the PriorityQueue data structure from util.py for UCS
    pQueue = util.PriorityQueue()  # Initialize priority queue for UCS
    pQueue.push((problem.getStartState(), []), 0)  # Push start state with cost 0
    visited = set()  # Track visited states

    while not pQueue.isEmpty():  # Continue while there are nodes in the queue
        currentNode, actions = pQueue.pop()  # Dequeue node with least total cost

        if problem.isGoalState(currentNode):  # Return actions if goal state is reached
            return actions

        if currentNode not in visited:  # Process if node hasn't been visited
            visited.add(currentNode)  # Mark node as visited

            for successor, action, stepCost in problem.getSuccessors(currentNode):  # Iterate over successors
                newActions = actions + [action]  # Calculate new actions
                newCost = problem.getCostOfActions(newActions)  # Calculate new cost
                pQueue.push((successor, newActions), newCost)  # Push successor with new cost to queue

    return []  # Return empty list if no solution found


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=util.manhattanDistance):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
