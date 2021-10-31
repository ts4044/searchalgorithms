#########################################
#                                       #
#                                       #
#  ==  SOKOBAN STUDENT AGENT CODE  ==   #
#                                       #
#      Written by: Tejas Sateesh        #
#                                       #
#                                       #
#########################################


# SOLVER CLASSES WHERE AGENT CODES GO
from helper import *
import random
import math


# Base class of agent (DO NOT TOUCH!)
class Agent:
    def getSolution(self, state, maxIterations):

        '''
        EXAMPLE USE FOR TREE SEARCH AGENT:


        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            [ POP NODE OFF OF QUEUE ]

            [ EVALUATE NODE AS WIN STATE]
                [ IF WIN STATE: BREAK AND RETURN NODE'S ACTION SEQUENCE]

            [ GET NODE'S CHILDREN ]

            [ ADD VALID CHILDREN TO QUEUE ]

            [ SAVE CURRENT BEST NODE ]


        '''


        '''
        EXAMPLE USE FOR EVOLUTION BASED AGENT:
        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            [ MUTATE ]

            [ EVALUATE ]
                [ IF WIN STATE: BREAK AND RETURN ]

            [ SAVE CURRENT BEST ]

        '''


        return []       # set of actions


#####       EXAMPLE AGENTS      #####

# Do Nothing Agent code - the laziest of the agents
class DoNothingAgent(Agent):
    def getSolution(self, state, maxIterations):
        if maxIterations == -1:     # RIP your machine if you remove this block
            return []

        #make idle action set
        nothActionSet = []
        for i in range(20):
            nothActionSet.append({"x":0,"y":0})

        return nothActionSet

# Random Agent code - completes random actions
class RandomAgent(Agent):
    def getSolution(self, state, maxIterations):

        #make random action set
        randActionSet = []
        for i in range(20):
            randActionSet.append(random.choice(directions))

        return randActionSet




#####    ASSIGNMENT 1 AGENTS    #####


# BFS Agent code
class BFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        # Initializing the best node to be the first node to avoid None Condition and an additional if in the loop
        bestNode = Node(state.clone(), None, None)
        queue = [Node(state.clone(), None, None)]
        visited = []

        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            # Get the node from the front of the queue
            current = queue.pop(0)

            # If the current node is not already visited, visit it
            if current.getHash() not in visited:
                visited.append(current.getHash())

                # Check if the current node is the win state
                if current.state.checkWin():
                    # If win, then return the actions to be taken to go to this state
                    return current.getActions()

                # If it is not the win state, then add the children to the queue to visit them
                children = current.getChildren()
                for child in children:
                    # Check if the child is already visited or not to avoid duplicates in the queue.
                    # Even if there are duplicates in queue, it won't matter
                    # But since we have limited iterations, we need this check
                    if child.getHash() not in visited:
                        queue.append(child)

                # For the best node, compare the heuristics with current before changing.
                # Even though this is an uninformed search the way bestNode is considered
                # is taking the closest node to goal state
                if current.getHeuristic() <= bestNode.getHeuristic():
                    bestNode = current

        # If no goal state was achieved, then return the closest node to the goal
        return bestNode.getActions()



# DFS Agent Code
class DFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        intializeDeadlocks(state)
        iterations = 0
        # Initializing the best node to be the first node to avoid None Condition and an additional if in the loop
        bestNode = Node(state.clone(), None, None)
        stack = [Node(state.clone(), None, None)]
        visited = []

        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(stack) > 0:
            iterations += 1

            # Get the node from the top of the stack
            current = stack.pop()

            # If the current node is not already visited, visit it
            if current.getHash() not in visited:
                visited.append(current.getHash())

                # Check if the current node is the win state
                if current.state.checkWin():
                    # If win, then return the actions to be taken to go to this state
                    return current.getActions()

                # If it is not the win state, then add the children to the stack to visit them
                # The difference between BFS and DFS is usage of the stack.
                # When we pop the stack, we are traversing down the tree cause we are visiting the node and its children
                # first and then going to the sibling.
                children = current.getChildren()
                for child in children:
                    # Check if the child is already visited or not to avoid duplicates in the stack.
                    # Even if there are duplicates in stack, it won't matter
                    # But since we have limited iterations, we need this check
                    if child.getHash() not in visited:
                        stack.append(child)

                # For the best node, compare the heuristics with current before changing.
                # Even though this is an uninformed search the way bestNode is considered
                # is taking the closest node to goal state
                if current.getHeuristic() <= bestNode.getHeuristic():
                    bestNode = current

        # If no goal state was achieved, then return the closest node to the goal
        return bestNode.getActions()



# AStar Agent Code
class AStarAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        # Initializing the best node to be the first node to avoid None Condition and an additional if in the loop
        bestNode = Node(state.clone(), None, None)

        #initialize priority queue
        queue = PriorityQueue()
        queue.put(Node(state.clone(), None, None))  # This is open
        visited = []  # This is closed

        while (iterations < maxIterations or maxIterations <= 0) and queue.qsize() > 0:
            iterations += 1

            # Deque and get the node with highest priority
            current = queue.get(False)

            # If the node is not visited, then visit it
            if current.getHash() not in visited:
                visited.append(current.getHash())

                # Check if the current node is the win state
                if current.state.checkWin():
                    # If win, then return the actions to be taken to go to this state
                    return current.getActions()

                # If it is not the goal state, get the children and add to priority queue
                # Since we are using a priority queue, we don't need to check the heuristics everytime
                # Since it is done internally. The PriorityQueue sorts the structure whenever a new element is inserted
                # Such that, the top of the queue is always the node with lowest heuristic.
                children = current.getChildren()
                for child in children:
                    if child.getHash() not in visited:
                        queue.put(child)

                # For the best node, compare the heuristics with current before changing.
                if current.getHeuristic() <= bestNode.getHeuristic():
                    bestNode = current

        # We don't usually reach here since A* gets to the solution almost always
        return bestNode.getActions()


#####    ASSIGNMENT 2 AGENTS    #####


# Hill Climber Agent code
class HillClimberAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0

        seqLen = 50            # maximum length of the sequences generated
        coinFlip = 0.5          # chance to mutate

        #initialize the first sequence (random movements)
        bestSeq = []
        for i in range(seqLen):
            bestSeq.append(random.choice(directions))
        bestSeqCopy = bestSeq.copy()

        #mutate the best sequence until the iterations runs out or a solution sequence is found
        while iterations < maxIterations:
            iterations += 1

            ## YOUR CODE HERE ##
            newstate = state.clone()

            for i in range(seqLen):
                newstate.update(bestSeq[i]['x'], bestSeq[i]['y'])

                if newstate.checkWin():
                    return bestSeq

            if getHeuristic(newstate) > getHeuristic(state):
                bestSeq = bestSeqCopy.copy()
            else:
                bestSeqCopy = bestSeq.copy()

            for i in range(seqLen):
                if random.random() < coinFlip:
                    bestSeq[i] = random.choice(directions)

        # return the best sequence found
        return bestSeq



# Genetic Algorithm code
class GeneticAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)

        iterations = 0
        seqLen = 50             # maximum length of the sequences generated
        popSize = 10            # size of the population to sample from
        parentRand = 0.5        # chance to select action from parent 1 (50/50)
        mutRand = 0.3           # chance to mutate offspring action

        bestSeq = []            #best sequence to use in case iterations max out

        #initialize the population with sequences of POP_SIZE actions (random movements)
        population = []
        for p in range(popSize):
            bestSeq = []
            for i in range(seqLen):
                bestSeq.append(random.choice(directions))
            population.append(bestSeq)

        total = popSize * (popSize + 1) / 2     # sum of first n numbers = n * (n+1) / 2
        weights = [i / total for i in reversed(range(1, popSize + 1))]

        #mutate until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations):
            iterations += 1

            # 1. evaluate the population
            heuristic_list = []
            for p in range(popSize):
                test_state = state.clone()
                for direction in population[p]:
                    test_state.update(direction['x'], direction['y'])

                    if test_state.checkWin():
                        return population[p]

                heuristic_list.append(getHeuristic(test_state))

            # 2. sort the population by fitness (low to high)
            evaluation_list = list(zip(heuristic_list, range(popSize)))
            evaluation_list.sort()

            # 2.1 save bestSeq from best evaluated sequence
            bestSeq = population[evaluation_list[0][1]]

            # 3. generate probabilities for parent selection based on fitness

            # 4. populate by crossover and mutation
            new_pop = []
            for i in range(int(popSize / 2)):
                selected_parents = random.choices(population=evaluation_list, weights=weights, k=2)

                # 4.1 select 2 parents sequences based on probabilities generated
                par1 = population[selected_parents[0][1]]
                par2 = population[selected_parents[1][1]]

                # 4.2 make a child from the crossover of the two parent sequences
                offspring = [par1[i] if random.random() < parentRand else par2[i] for i in range(seqLen)]

                # 4.3 mutate the child's actions
                offspring = [random.choice(directions) if random.random() < mutRand else offspring[i] for i in
                             range(seqLen)]

                # 4.4 add the child to the new population
                new_pop.append(list(offspring))

            # 5. add top half from last population (mu + lambda)
            for i in range(int(popSize / 2)):
                new_pop.append(population[evaluation_list[i][1]])

            # 6. replace the old population with the new one
            population = list(new_pop)

        # return the best found sequence
        return bestSeq


# MCTS Specific node to keep track of rollout and score
class MCTSNode(Node):
    def __init__(self, state, parent, action, maxDist):
        super().__init__(state,parent,action)
        self.children = []  #keep track of child nodes
        self.n = 0          #visits
        self.q = 0          #score
        self.maxDist = maxDist      #starting distance from the goal (heurstic score of initNode)

    #update get children for the MCTS
    def getChildren(self,visited):
        #if the children have already been made use them
        if(len(self.children) > 0):
            return self.children

        children = []

        #check every possible movement direction to create another child
        for d in directions:
            childState = self.state.clone()
            crateMove = childState.update(d["x"], d["y"])

            #if the node is the same spot as the parent, skip
            if childState.player["x"] == self.state.player["x"] and childState.player["y"] == self.state.player["y"]:
                continue

            #if this node causes the game to be unsolvable (i.e. putting crate in a corner), skip
            if crateMove and checkDeadlock(childState):
                continue

            #if this node has already been visited (same placement of player and crates as another seen node), skip
            if getHash(childState) in visited:
                continue

            #otherwise add the node as a child
            children.append(MCTSNode(childState, self, d, self.maxDist))

        self.children = list(children)    #save node children to generated child

        return children

    #calculates the score the distance from the starting point to the ending point (closer = better = larger number)
    def calcEvalScore(self,state):
        return self.maxDist - getHeuristic(state)

    #compares the score of 2 mcts nodes
    def __lt__(self, other):
        return self.q < other.q

    #print the score, node depth, and actions leading to it
    #for use with debugging
    def __str__(self):
        return str(self.q) + ", " + str(self.n) + ' - ' + str(self.getActions())


# Monte Carlo Tree Search Algorithm code
class MCTSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        #setup
        intializeDeadlocks(state)
        iterations = 0
        bestNode = None
        initNode = MCTSNode(state.clone(), None, None, getHeuristic(state))

        while(iterations < maxIterations):
            #print("\n\n---------------- ITERATION " + str(iterations+1) + " ----------------------\n\n")
            iterations += 1

            #mcts algorithm
            rollNode = self.treePolicy(initNode)
            score = self.rollout(rollNode)
            self.backpropogation(rollNode, score)

            #if in a win state, return the sequence
            if(rollNode.checkWin()):
                return rollNode.getActions()

            #set current best node
            bestNode = self.bestChildUCT(initNode)

            #if in a win state, return the sequence
            if(bestNode and bestNode.checkWin()):
                return bestNode.getActions()


        #return solution of highest scoring descendent for best node
        #if this line was reached, that means the iterations timed out before a solution was found
        return self.bestActions(bestNode)


    #returns the descendent with the best action sequence based
    def bestActions(self, node):
        #no node given - return nothing
        if node == None:
            return []

        bestActionSeq = []
        while(len(node.children) > 0):
            node = self.bestChildUCT(node)

        return node.getActions()


    ####  MCTS SPECIFIC FUNCTIONS BELOW  ####

    #determines which node to expand next
    def treePolicy(self, rootNode):
        curNode = rootNode
        visited = []

        ## YOUR CODE HERE ##
        while curNode.checkWin() is False:
            children = curNode.getChildren(visited)
            if len(children) == 0:
                break

            for child in children:
                if child.n == 0:
                    return child

            curNode = self.bestChildUCT(curNode)

        return curNode



    # uses the exploitation/exploration algorithm
    def bestChildUCT(self, node):
        c = 1               #c value in the exploration/exploitation equation
        bestChild = None

        ## YOUR CODE HERE ##
        child_values = []
        children = node.getChildren([])

        numerator = 2 * math.log(node.n)

        if len(children) != 0:
            for child in children:
                if child.n == 0:
                    continue
                else:
                    score = (child.q / child.n) + (c * math.sqrt(numerator / child.n))
                    child_values.append((score, child))

            child_values.sort()
            bestChild = child_values[-1][1]

        return bestChild



     #simulates a score based on random actions taken
    def rollout(self,node):
        numRolls = 7        #number of times to rollout to

        ## YOUR CODE HERE ##
        if node.state.checkWin():
            return node.calcEvalScore(node.state)

        state_clone = node.state.clone()
        for i in range(numRolls):
            random_action = random.choice(directions)
            state_clone.update(random_action['x'], random_action['y'])
            if state_clone.checkWin():
                break
        return node.calcEvalScore(state_clone)



     #updates the score all the way up to the root node
    def backpropogation(self, node, score):

        ## YOUR CODE HERE ##
        while node is not None:
            node.n += 1
            node.q = node.q + score
            node = node.parent

        return


