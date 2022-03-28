from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from builtins import range
from builtins import object
import util
import copy
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."
    old_lines = ""

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.printInfo(gameState)
        return KeyboardAgent.getAction(self, gameState)

    def printInfo(self, gameState):
        print("---------------- TICK --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())


    # noinspection PyStatementEffect
    def printLineData(self, gameState):
        
        #subconjunto_1 info of pacman and legal moves
        #subconjunto_2 info of ghosts and distance of the food (conclusion target of the pacman)

        linedata = str(gameState.getPacmanPosition()[0]) + ","+str(gameState.getPacmanPosition()[1]) + ","

        # Legal actions
        legal_actions = gameState.getLegalPacmanActions()
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        for action in actions:
            linedata += self.found_direction(direction=action,list=legal_actions)


        for i, ghost in enumerate(gameState.getGhostPositions()):
            if (gameState.getLivingGhosts()[i+1]):
                linedata+=str(0)+","+str(ghost[0]) +","+str(ghost[1]) +","
            else:
                linedata+=str(1)+","+str(ghost[0]) +","+str(ghost[1]) +","

        if gameState.getDistanceNearestFood() != None:
            linedata += str(gameState.getDistanceNearestFood()) + ","  # Distance nearest pac dots
        if gameState.getDistanceNearestFood() == None:
            linedata += str(0) + ","

        linedata += str(gameState.getScore())+","

        linedata += str(KeyboardAgent.getAction(self, gameState)) + ","

        if self.old_lines!="":
            self.old_lines +=str(gameState.getScore())+","
            self.old_lines +=str(KeyboardAgent.getAction(self, gameState)) + "\n"
            with open("training_keyboard.arff", "a") as output_file:
                output_file.write(self.old_lines)
            self.old_lines = linedata
            return "XXXXXXXXXX"
        self.old_lines = linedata
        return "XXXXXXXXXX"

    def found_direction(self, direction, list):
        for move in list:
            if move==direction:
                return "1,"
        return "0,"

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):
    old_line=""
    move=None

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        self.old_num_dots = gameState.getNumFood()

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)

        # Decides new move
        move = self.bfs(gameState)
        print("-- Next move: ",move,"--")
        self.move=move
        return move

    def bfs(self, gameState):
        if not any(gameState.getLivingGhosts()):
            return Directions.STOP
        end_states = []
        found_length = 100000
        expanded_nodes = []
        empty_list = []
        next_nodes = [[gameState.getPacmanPosition()[0],gameState.getPacmanPosition()[1],empty_list,gameState.getScore()]]
        while len(next_nodes)>0:
            if (len(next_nodes[0][2])<=found_length):
                # generate succesor
                succesors = self.generate_succesors(gameState, expanded_nodes, next_nodes[0])

                # check if expanded is goal
                if self.is_goal(gameState, next_nodes[0]):
                    end_states.append(next_nodes[0])
                    found_length = len(next_nodes[0][2])
                else:
                    # add successors to list
                    next_nodes = self.add_succesors(next_nodes, succesors)

                # remove expanded node
                expanded_nodes.append(next_nodes.pop(0))
            else:
                break
        return self.choose_step(end_states)

    def add_succesors(self, next_nodes, succesors):
        for suc in succesors:
            found = self.is_in(next_nodes, suc)
            if found!=-1:
                if suc[3]>next_nodes[found][3]:
                    next_nodes[found]=suc
            else:
                next_nodes.append(suc)
        return next_nodes

    def choose_step(self,end_states):
        chosen = None
        max = -100000
        for state in end_states:
            if state[3]>max:
                chosen=state
                max=state[3]

        return chosen[2][0]

    def is_goal(self, gameState, pos):
        for i, ghost in enumerate(gameState.getGhostPositions()):
            if ghost[0]==pos[0] and ghost[1]==pos[1] and gameState.getLivingGhosts()[i+1]:
                return True
        return False

    def is_in(self, list, state):
        for i, item in enumerate(list):
            if item[0]==state[0] and item[1]==state[1]:
                return i
        return -1

    def check_food(self, gameState, posx, posy):
        if gameState.getFood()[posx][posy]==True:
            return 100
        return 0

    def generate_succesors(self, gameState, expanded_nodes, position):
        succesors = []
        next_score = position[3]-1
        if gameState.getWalls()[position[0]+1][position[1]]==False:
            pos = copy.deepcopy(position[2])
            pos.append(Directions.EAST)
            score = next_score + self.check_food(gameState, position[0]+1,position[1])
            succesors.append([position[0]+1,position[1],pos,score])
        if gameState.getWalls()[position[0]-1][position[1]]==False:
            pos = copy.deepcopy(position[2])
            pos.append(Directions.WEST)
            score = next_score + self.check_food(gameState, position[0]-1,position[1])
            succesors.append([position[0]-1,position[1],pos,score])
        if gameState.getWalls()[position[0]][position[1]+1]==False:
            pos = copy.deepcopy(position[2])
            pos.append(Directions.NORTH)
            score = next_score + self.check_food(gameState, position[0],position[1]+1)
            succesors.append([position[0],position[1]+1,pos,score])
        if gameState.getWalls()[position[0]][position[1]-1]==False:
            pos = copy.deepcopy(position[2])
            pos.append(Directions.SOUTH)
            score = next_score + self.check_food(gameState, position[0],position[1]-1)
            succesors.append([position[0],position[1]-1,pos,score])
        aux = []
        for suc in succesors:
            if self.is_in(expanded_nodes, suc)==-1:
                aux.append(suc)
        return aux

    def printLineData(self, gameState):

        # subconjunto_1 info of pacman and legal moves
        # subconjunto_2 info of ghosts and distance of the food (conclusion target of the pacman)

        linedata = str(gameState.getPacmanPosition()[0]) + "," + str(gameState.getPacmanPosition()[1]) + ","

        # Legal actions
        legal_actions = gameState.getLegalPacmanActions()
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        for action in actions:
            linedata += self.found_direction(direction=action, list=legal_actions)

        for i, ghost in enumerate(gameState.getGhostPositions()):
            if (gameState.getLivingGhosts()[i + 1]):
                linedata += str(0) + "," + str(ghost[0]) + "," + str(ghost[1]) + ","
            else:
                linedata += str(1) + "," + str(ghost[0]) + "," + str(ghost[1]) + ","

        if gameState.getDistanceNearestFood() != None:
            linedata += str(gameState.getDistanceNearestFood()) + ","  # Distance nearest pac dots
        if gameState.getDistanceNearestFood() == None:
            linedata += str(0) + ","

        linedata += str(gameState.getScore()) + ","

        linedata += str(self.move) + ","

        if self.old_line != "":
            self.old_line += str(gameState.getScore()) + ","
            self.old_line += str(self.move) + "\n"
            with open("training_tutorial1.arff", "a") as output_file:
                output_file.write(self.old_line)
            self.old_line = linedata
            return "XXXXXXXXXX"
        self.old_line = linedata
        return "XXXXXXXXXX"

    def found_direction(self, direction, list):
        for move in list:
            if move == direction:
                return "1,"
        return "0,"