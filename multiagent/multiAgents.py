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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newCapsules = successorGameState.getCapsules()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # try using least distance of eating all food.
        newFoodList = newFood.asList()
        Score = 0
        newGhostPos = successorGameState.getGhostPositions()
        nearestGhostDis = min([manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos])
        if nearestGhostDis <= 5: # 进入戒备状态
            Score += nearestGhostDis * 3
        else:
            Score += nearestGhostDis + 21 # this bias is important, avoid my agent trying to be 若即若离 with ghost.
        # if len(newCapsules) > 0:
        #     Score -= manhattanDistance(newCapsules[0], newPos) * 1000 # I dont need such thing
        s = 0
        for _ in range(len(newFoodList)):
            tmp, pos = min([(util.manhattanDistance(newPos, food), food) for food in newFoodList])
            newFoodList.remove(pos)
            if _ == 0:
                s += tmp # 可以*3 encourage吃掉最近的豆子
            else:
                s += tmp
            newPos = pos
            # print(s)
        Score -= s * 2
        # Score -= len(newFoodList)
        
        successorGameState.data.score = Score
        # print(Score)
        # print(len(newFoodList))
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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


    def evaluationFunction1(self, currentGameState: GameState, agentIndex):
        if currentGameState.isWin():
            return 1
        elif currentGameState.isLose():
            return 0
        else:
            legalActions = GameState.getLegalActions(agentIndex=agentIndex)
            successorGameStates = [GameState.generateSuccessor(agentIndex=agentIndex, action=legalAction) for legalAction in legalActions]
            scores = [self.evaluationFunction1(successorGameState) for successorGameState in successorGameStates]
            if agentIndex == 0:
                bestScore = max(scores)
            else:
                bestScore = min(scores)
            return bestScore


    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def term(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        ghostIndexs = list(range(1, gameState.getNumAgents()))
        def minValue(state, depth, ghostIndex):
            if term(state, depth):
                return self.evaluationFunction(state)
            value = 1e9
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostIndexs[-1]:
                    value = min(value, maxValue(state.generateSuccessor(ghostIndex, action), depth + 1))
                else:
                    value = min(value, minValue(state.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1))
            return value
        
        def maxValue(state, depth):
            if term(state, depth):
                return self.evaluationFunction(state)
            value = -1e9
            for action in state.getLegalActions(0):
                value = max(value, minValue(state.generateSuccessor(0, action), depth, ghostIndexs[0]))
            return value
        # Collect legal moves and successor states
        res = [(action, minValue(gameState.generateSuccessor(0, action), 0, ghostIndexs[0])) for action in gameState.getLegalActions(agentIndex=0)]
        res.sort(key=lambda x: x[1], reverse=True)
        return res[0][0]
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def term(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        ghostIndexs = list(range(1, gameState.getNumAgents()))
        def minValue(state, depth, ghostIndex, alpha, beta):
            if term(state, depth):
                return self.evaluationFunction(state)
            value = 1e9
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostIndexs[-1]:
                    value = min(value, maxValue(state.generateSuccessor(ghostIndex, action), depth + 1, alpha, beta))
                else:
                    value = min(value, minValue(state.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1, alpha, beta))
                if value < alpha: return value
                beta = min(beta, value)
            return value
        
        def maxValue(state, depth, alpha, beta):
            if term(state, depth):
                return self.evaluationFunction(state)
            value = -1e9
            for action in state.getLegalActions(0):
                value = max(value, minValue(state.generateSuccessor(0, action), depth, ghostIndexs[0], alpha, beta))
                if value > beta: return value
                alpha = max(alpha, value)
            return value
        # Collect legal moves and successor states
        alpha = -1e9
        beta = 1e9
        curValue = -1e9
        curAction = Directions.STOP
        for nextAction in gameState.getLegalActions(0).copy():
            nextState = gameState.generateSuccessor(0, nextAction)
            nextValue = minValue(nextState, 0, 1, alpha, beta)

            if nextValue > curValue:
                curValue, curAction = nextValue, nextAction
            alpha = max(alpha, curValue)
        
        return curAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def term(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        ghostIndexs = list(range(1, gameState.getNumAgents()))
        def randomValue(state, depth, ghostIndex):
            value = []
            if term(state, depth):
                return self.evaluationFunction(state)
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostIndexs[-1]:
                    value.append(maxValue(state.generateSuccessor(ghostIndex, action), depth + 1))
                else:
                    value.append(randomValue(state.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1))

            return sum(value) / len(value)
        
        def maxValue(state, depth):
            if term(state, depth):
                return self.evaluationFunction(state)
            value = -1e9
            for action in state.getLegalActions(0):
                value = max(value, randomValue(state.generateSuccessor(0, action), depth, ghostIndexs[0]))
            return value
        # Collect legal moves and successor states
        curValue = -1e9
        curAction = Directions.STOP
        for nextAction in gameState.getLegalActions(0).copy():
            nextState = gameState.generateSuccessor(0, nextAction)
            nextValue = randomValue(nextState, 0, 1)

            if nextValue > curValue:
                curValue, curAction = nextValue, nextAction
        
        return curAction
        

def betterEvaluationFunction(GameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = GameState.getPacmanPosition()
    Food = GameState.getFood()
    GhostStates = GameState.getGhostStates()
    Capsules = GameState.getCapsules()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    FoodList = Food.asList()
    # FoodList.extend(Capsules)
    Score = 0
    GhostPos = GameState.getGhostPositions()
    nearestGhostDis = min([manhattanDistance(newPos, ghostPos) for ghostPos in GhostPos])
    if nearestGhostDis <= 5: # 进入戒备状态
        Score += nearestGhostDis * 3
    else:
        Score += nearestGhostDis + 21 # this bias is important, avoid my agent trying to be 若即若离 with ghost.
            # if len(newCapsules) > 0:
            #     Score -= manhattanDistance(newCapsules[0], newPos) * 1000 # I dont need such thing
    s = 0
    if len(FoodList) > 0:
        # Score -= min([util.manhattanDistance(newPos, food) for food in FoodList]) * 10
        for _ in range(len(FoodList)):
            tmp, pos = min([(util.manhattanDistance(newPos, food), food) for food in FoodList]) 
            FoodList.remove(pos)
            s += tmp #encourage吃掉最近的豆子
            newPos = pos
        # print(s)
        Score -= s * 2
    
    
        
    GameState.data.score = Score

    return GameState.getScore()
    

# Abbreviation
better = betterEvaluationFunction
