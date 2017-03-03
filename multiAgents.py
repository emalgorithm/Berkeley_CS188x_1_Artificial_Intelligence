# multiAgents.py
# --------------
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
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()

        # Get distance to closest food
        newFoodList = newFood.asList()
        closestFoodDist = sys.maxint

        for food in newFoodList:
            dist = manhattanDistance(newPos, food)
            closestFoodDist = min(closestFoodDist, dist)

        # Get distance to closest ghost
        ghostsPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        closestGhostDist = sys.maxint

        for ghost in ghostsPositions:
            dist = manhattanDistance(newPos, ghost)
            closestGhostDist = min(closestGhostDist, dist)

        if closestGhostDist <= 1:
            return -1000

        if score > currentGameState.getScore() and closestGhostDist > 2:
            return 1000

        return score + closestGhostDist - 3 * closestFoodDist


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # Return a tuple (bestAction, bestValue) for an agent from gameState according to minimax
        # with the given depth
        def value(gameState, agentIndex, depth):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth -= 1

            if depth == 0 or not gameState.getLegalActions(agentIndex):
                return None, self.evaluationFunction(gameState)

            # If the agent is pacman (agentIndex equals to 0), then maximize
            if agentIndex == 0:
                return maximize(gameState, agentIndex, depth)
            # If the agent is a ghost, then minimize
            else:
                return minimize(gameState, agentIndex, depth)

        # Return a tuple (maxAction, maxValue) for a maximizing agent from gameState according to
        # minimax with the given depth
        def maximize(gameState, agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            maxValue = -sys.maxint - 1
            maxAction = None

            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                _, v = value(successorGameState, agentIndex + 1, depth)
                if v > maxValue:
                    maxValue = v
                    maxAction = action

            return maxAction, maxValue

        # Return a tuple (minAction, minValue) for a minimizing agent from gameState according to
        # minimax with the given depth
        def minimize(gameState, agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            minValue = sys.maxint
            minAction = None

            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                _, v = value(successorGameState, agentIndex + 1, depth)
                if v < minValue:
                    minValue = v
                    minAction = action

            return minAction, minValue


        return value(gameState, 0, self.depth)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Return a tuple (bestAction, bestValue) for an agent from gameState according to minimax
        # with the given depth
        def value(gameState, agentIndex, depth, alpha, beta):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth -= 1

            if depth == 0 or not gameState.getLegalActions(agentIndex):
                return None, self.evaluationFunction(gameState)

            # If the agent is pacman (agentIndex equals to 0), then maximize
            if agentIndex == 0:
                return maximize(gameState, agentIndex, depth, alpha, beta)
            # If the agent is a ghost, then minimize
            else:
                return minimize(gameState, agentIndex, depth, alpha, beta)

        # Return a tuple (maxAction, maxValue) for a maximizing agent from gameState according to
        # minimax with the given depth
        def maximize(gameState, agentIndex, depth, alpha, beta):
            actions = gameState.getLegalActions(agentIndex)
            maxValue = -sys.maxint - 1
            maxAction = None

            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                _, v = value(successorGameState, agentIndex + 1, depth, alpha, beta)
                if v > maxValue:
                    maxValue = v
                    maxAction = action
                    if v > beta:
                        break
                    alpha = max(alpha, v)

            return maxAction, maxValue

        # Return a tuple (minAction, minValue) for a minimizing agent from gameState according to
        # minimax with the given depth
        def minimize(gameState, agentIndex, depth, alpha, beta):
            actions = gameState.getLegalActions(agentIndex)
            minValue = sys.maxint
            minAction = None

            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                _, v = value(successorGameState, agentIndex + 1, depth, alpha, beta)
                if v < minValue:
                    minValue = v
                    minAction = action
                    if v < alpha:
                        break
                    beta = min(beta, v)

            return minAction, minValue


        return value(gameState, 0, self.depth, -sys.maxint - 1, sys.maxint)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # Return a tuple (bestAction, bestValue) for an agent from gameState according to minimax
        # with the given depth
        def value(gameState, agentIndex, depth):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth -= 1

            if depth == 0 or not gameState.getLegalActions(agentIndex):
                return None, self.evaluationFunction(gameState)

            # If the agent is pacman (agentIndex equals to 0), then maximize
            if agentIndex == 0:
                return maximize(gameState, agentIndex, depth)
            # If the agent is a ghost, then minimize
            else:
                return None, minimize(gameState, agentIndex, depth)

        # Return a tuple (maxAction, maxValue) for a maximizing agent from gameState according to
        # expectimax with the given depth
        def maximize(gameState, agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            maxValue = -sys.maxint - 1
            maxAction = None

            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                _, v = value(successorGameState, agentIndex + 1, depth)
                if v > maxValue:
                    maxValue = v
                    maxAction = action

            return maxAction, maxValue

        # Return the expected value for a minimizing agent from gameState according to
        # expectimax
        def minimize(gameState, agentIndex, depth):
            actions = gameState.getLegalActions(agentIndex)
            expectedValue = 0.0
            expectedAction = None

            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                _, v = value(successorGameState, agentIndex + 1, depth)
                expectedValue += (1.0 / len(actions)) * v

            return expectedValue


        return value(gameState, 0, self.depth)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

