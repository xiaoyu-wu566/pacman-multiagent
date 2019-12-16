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

        "*** YOUR CODE HERE ***"
        oldFood = currentGameState.getFood();

        totalScore = 0.0
        for ghost in newGhostStates:
            d = manhattanDistance(ghost.getPosition(), newPos)
            if ghost.scaredTimer > 0:
                totalScore += 1000 * ghost.scaredTimer/(d+1)
            else:
                totalScore -= 100/(d+1)

        ds = sorted([manhattanDistance(food, newPos) for food in oldFood.asList()])
        for d in ds:
            totalScore += 50/(d+1)

        return totalScore

        # return successorGameState.getScore()

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

    def minimax_fun(self, agent, depth, gameState):

        if depth >= self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # pacman if agent == 0
        if agent == 0:
            return max([self.minimax_fun(1, depth, gameState.generateSuccessor(agent, nextAction)) for nextAction in
                        gameState.getLegalActions(agent)])
            # current_agent += 1
        else:
            # current_agent += 1
            if agent == gameState.getNumAgents()-1:
                nextAgent = 0
                depth += 1
            else:
                nextAgent = agent + 1
            return min(
                [self.minimax_fun(nextAgent, depth, gameState.generateSuccessor(agent, nextAction)) for nextAction in
                 gameState.getLegalActions(agent)])


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

        scores = []
        actions = []

        for nextAction in gameState.getLegalActions(0):
            scores.append(self.minimax_fun(1, 0, gameState.generateSuccessor(0, nextAction)))
            actions.append(nextAction)

        best_action = actions[0]
        best_score = scores[0]
        for i in range(1, len(scores)):
            if scores[i] > best_score:
                best_score = scores[i]
                best_action = actions[i]
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def max_value(agent, depth, game_state, alpha, beta):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)

            v = float("-inf")
            for newAction in game_state.getLegalActions(agent):
                v = max(v, value(1, depth, game_state.generateSuccessor(agent, newAction), alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(agent, depth, game_state, alpha, beta):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)

            v = float("inf")

            next_agent = agent + 1
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
                depth += 1

            for newAction in game_state.getLegalActions(agent):
                v = min(v, value(next_agent, depth, game_state.generateSuccessor(agent, newAction), alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(agent, depth, game_state, alpha, beta):
            if agent == 0:
                return max_value(agent, depth, game_state, alpha, beta)
            else:
                return min_value(agent, depth, game_state, alpha, beta)

        utility = float("-inf")
        action =Directions.STOP
        alpha = float("-inf")
        beta= float("inf")
        # for action in gameState.getLegalActions(0):
        #     nextAction = gameState.generatePacmanSuccessor(0, action)
        #     ghost = utility
        #     utility = max(utility, min_value(1, self.depth,nextAction,alpha,beta))

        for nextAction in gameState.getLegalActions(0):
            ghost = value(1, 0, gameState.generateSuccessor(0, nextAction), alpha, beta)
            if ghost > utility:
                utility = ghost
                action = nextAction
            alpha = max(alpha, utility)

        return action


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

        def expectimax_fun(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            n_actions = len(gameState.getLegalActions(agent))
            if agent == 0:
                return max(expectimax_fun(1, depth, gameState.generateSuccessor(agent, newAction))
                           for newAction in gameState.getLegalActions(agent))
            else:
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                    depth += 1
                return sum(expectimax_fun(nextAgent, depth, gameState.generateSuccessor(agent, newAction))
                           for newAction in gameState.getLegalActions(agent)) / float(n_actions)

        maximum = float("-inf")
        action = Directions.STOP
        for nextAction in gameState.getLegalActions(0):
            utility = expectimax_fun(1, 0, gameState.generateSuccessor(0, nextAction))
            if utility > maximum:
                maximum = utility
                action = nextAction

        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <consider the distance between pacman and foods and pacman and ghosts,
                    fine-tune the coefficients>
    """
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    pos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostPositions()
    foods = currentGameState.getFood()

    max_dis = 1e10
    tol = 1e-4

    # food distance
    food_distances = [manhattanDistance(food, pos) for food in foods.asList()]
    if len(food_distances) > 0:
        food_distance_min = min(food_distances)
        food_distance_mean = sum(food_distances)/float(len(food_distances))

    else:
        food_distance_min = max_dis
        food_distance_mean= max_dis
    ghost_distances = [manhattanDistance(ghost, pos) for ghost in ghosts]
    if len(ghost_distances) > 0:

        ghost_distance_mean = sum(ghost_distances)/float(len(ghost_distances))
        ghost_distance_min = min(ghost_distances)

    else:
        ghost_distance_mean = max_dis
        ghost_distance_min = max_dis
    return currentGameState.getScore() + 10/(food_distance_min + tol) + 10/(food_distance_mean+tol) \
           - 1/(ghost_distance_min+tol) - 1/(ghost_distance_mean + tol)


# Abbreviation
better = betterEvaluationFunction

