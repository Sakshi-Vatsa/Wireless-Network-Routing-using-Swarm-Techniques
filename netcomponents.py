#!/usr/bin/env
import random
import pdb
import numpy as np
import random as rn
#import networkx as nx
from numpy.random import choice as np_choice
import matplotlib.pyplot as plt
import netmodel as model
import os
import copy
import time

NUMBER_OF_NODE=10
vertices=NUMBER_OF_NODE
DEBUG = 0
POWER_DEPLETE = 0.0003

INFINITY = 4096
infinity= 4096
W_D = 0.3
W_L = 0.3
W_C = 0.1
W_P = 0.3
res=[]
NODE_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
NODE_RELIABILITY = {}
NODE_RELIABILITY["A"] = 0.91
NODE_RELIABILITY["B"] = 0.63
NODE_RELIABILITY["C"] = 0.66
NODE_RELIABILITY["D"] = 0.54
NODE_RELIABILITY["E"] = 0.71
NODE_RELIABILITY["F"] = 0.80
NODE_RELIABILITY["G"] = 0.71
NODE_RELIABILITY["H"] = 0.92
NODE_RELIABILITY["I"] = 0.81

# Set up sink
sink=9
SINK = "SINK"
SINK_RELIABILITY = 1.0

# Set up edges and distances along those edges
EDGE_DICT = {}
EDGE_DICT["A"] = [(SINK, 20), ("B", 8), ("C", 32), ("D", 17), ("E", 17), ("F", 17)]
EDGE_DICT["B"] = [(SINK, 8), ("A", 17), ("G", 39), ("C", 12)]
EDGE_DICT["C"] = [(SINK, 48), ("A", 32), ("B", 12), ("D", 8), ("G", 18), ("H", 28)]
EDGE_DICT["D"] = [(SINK, 58), ("A", 17), ("C", 8), ("E", 17), ("H", 30), ("I", 24)]
EDGE_DICT["E"] = [("A", 17), ("D", 17), ("F", 4), ("H", 38), ("I", 12)]
EDGE_DICT["F"] = [("A", 17), ("E", 4), ("I", 12)]
EDGE_DICT["G"] = [("B", 39), ("C", 18), ("H", 12)]
EDGE_DICT["H"] = [("C", 28), ("D", 30), ("E", 38), ("G", 12), ("I", 12)]
EDGE_DICT["I"] = [("D", 24), ("E", 12), ("F", 12), ("H", 12)]
EDGE_DICT["SINK"] = []

SINK_BUFFER_SIZE = 20000
NODE_BUFFER_SIZE = 2

MESSAGE_SEND_RATE = float(1.75 / len(NODE_LIST))
STEPS = 2000
COOL_DOWN = 200

# W_D = 0.0
# W_L = 0.0
# W_C = 1.0
# W_P = 0.0

# W_D = 0.25
# W_L = 0.25
# W_C = 0.25
# W_P = 0.25

Q_DEPTH = 2
DISCOUNT_RATE = 0.5

# DETERMINISTIC (FOR VALIDATION) : LEARNING = False (RUN INDIVIDUAL OPTIMIZATIONS HERE)
# LEARNING : BETA <= 0.5 (= 0.3)
# ... LEARNING WITH REWARD HISTORY: ALPHA_RATE > 0
# ...... COMPLETE REMEMBRANCE (ALL HISTORY IS VIEWED THE SAME): ALPHA_RATE = 1.0
# ...... SLOW-DECAYING REMEMBRANCE: ALPHA_RATE = 0.8
# ...... WITHOUT REWARD HISTORY (ALPHA_RATE = 0.0)
# REINFORCEMENT LEARNING : ZETA < 1.0 (= 0.75)
# ... COMPLETE REINFORCEMENT LEARNING  : BETA_RATE = 1.0
# ...... DECAYING REMEMBRANCE: ALPHA_RATE = 0.5
# ...... NO REMEMBRANCE: ALPHA_RATE = 0
# ... HYBRID REINFORCEMENT LEARNING : BETA_RATE <= 0.5 (= 0.3)
# ...... DECAYING REMEMBRANCE: ALPHA_RATE = 0.5
# ...... NO REMEMBRANCE: ALPHA_RATE = 0

ALPHA_RATE = 1.0
BETA_RATE = 0.3
ZETA_RATE = 0.0
HISTORY_WINDOW = 16
MAX_LEARNED_REWARD = sum([ALPHA_RATE**i for i in range(HISTORY_WINDOW)])

LEARNING = True # Turn off for naive

def calculate_least_adjacent_cost(adjacency_list, i, v, cache):
    adjacent_nodes = adjacency_list[v]
    least_adjacent_cost = float("inf")
    for node in adjacent_nodes:
      adjacent_cost = cache[node["from"]-1] + node["weight"]
      if adjacent_cost < least_adjacent_cost:
        least_adjacent_cost = adjacent_cost
    return least_adjacent_cost

    
# Debug command for development
def debug(flag, message):
    if flag:
        print (message)

# Basic message class
class Message:
    DELIVERED = 0
    SUCCESS = 1
    FAIL = 2
    def __init__(self, src, dest, data, ttl):
        self.src = src
        self.dest = dest
        self.data = data
        self.path = [src]
        # TTL is subtracted from at each hop
        self.ttl = ttl

class sw(object):

    def __init__(self):

        self.__Positions = []
        self.__Gbest = []

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""

        return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""

        return list(self.__Gbest)

class pso(sw):
    """
    Particle Swarm Optimization
    """
    def costFunction(self,pr):
      cost=0;
      for sr in NODE_LIST:
        tem=sr
        vis=set({})
        curr_path_cost=0
        distance_costs_per_node=[]
        #print("starting from %d"%tem)
        while (True):
          if(tem==SINK):
            cost+=curr_path_cost
            distance_costs_per_node.append(curr_path_cost)
          if(tem==""):
            cost+=infinity
            break
          max=("",-1,infinity);#(node,priority,weight)
          vis.add(tem)
          for x in EDGE_DICT[sr]:
            if (x[0]==SINK):
              place=sink
            else:
              place=ord(x[0])-65
            if pr[place]>max[1] and x[0] not in vis:
              max=(x[0],pr[place],x[1])
          tem=max[0]
          curr_path_cost+=max[2]
      return cost

    def __init__(self, n, function, lb, ub, dimension, iteration, w=0.5, c1=1,c2=1):
        super(pso, self).__init__()

        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        velocity = np.zeros((n, dimension))
        self._points(self.__agents)

        Pbest = self.__agents[np.array([function(x) for x in self.__agents]).argmin()]
        Gbest = Pbest

        for t in range(iteration):

            r1 = np.random.random((n, dimension))
            r2 = np.random.random((n, dimension))
            velocity = w * velocity + c1 * r1 * (Pbest - self.__agents) + c2 * r2 * (Gbest - self.__agents)
            self.__agents += velocity
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)

            Pbest = self.__agents[np.array([function(x) for x in self.__agents]).argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest
            res.append(function(Gbest))

        self._set_Gbest(Gbest)
# Basic neighbor class
class Connection:
    def __init__(self, thisNode, thatNode, distance):
        self.node = thatNode
        self.reliability = (thisNode.reliability + thatNode.reliability) / 2
        self.distance = distance

# Basic sensor node class
class SensorNode:
    def __init__(self, name, adjDict, bufferSize, reliability, power=1, powerDeplete=POWER_DEPLETE):
        self.name = name
        self.sinkName = "SINK"
        # Neighbor initialization from adjacency list
        self.neighbors = []
        # Calculate distance using Djikstra's for shortest number of hops
        self.distance = INFINITY
        self.maxDistance = 1
        # Assign power
        self.power = power
        self.powerDeplete = powerDeplete
        # Assign reliability
        self.reliability = reliability
        # Set up buffer
        self.rxBuffer = []
        self.txBuffer = []
        self.maxBufferSize = bufferSize
        self.congestion = 0
        # Maintain learned reward history
        self.histories = {}
        self.history = [1] * HISTORY_WINDOW
        # Set up Q-value
        self.Q = 2 if self.name == self.sinkName else 0
        if LEARNING:
            self.Q = {}

    # Report stats. Returns string.
    def generate_stats(self, distanceWanted=False):
        stats = ("Node %s - PWR: %f | CON: %f | RX: %d | TX: %d" \
            % (self.name, self.power, self.congestion, \
                len(self.rxBuffer), len(self.txBuffer)))
        if distanceWanted:
            stats = stats + " | DST: %d" % (self.distance)
        return stats

    # Node update per step (NOT COMPLETE)
    def update_node(self):
        UP_DEBUG = 1
        # If node has no power, it is dead
        if self.power <= 0:
            debug(UP_DEBUG, "UP: Node %s is dead. Cannot update." % (self.name))
            return
        debug(UP_DEBUG, "UP: Updating %s..." % (self.name))
        # Update congestion information
        self.congestion = self.get_congestion()
        # Deplete power
        self.consume_power()
        # Generate Q value
        if LEARNING:
            self.Q = self.generate_Q_value_plus()
        else:
            self.Q = self.generate_Q_value_naive()
        # Report stats for everything
        debug(UP_DEBUG, self.generate_stats())
        # for n in self.neighbors:
        #     debug(UP_DEBUG, "\t%s :: %s" % (n.node.name, str(self.histories[n.node.name])))

        # Process buffers
        RX_DEBUG = 1
        if len(self.rxBuffer):
            m = self.rxBuffer.pop(0)
            # print(m.data, " ", m.ttl)
            if m.ttl <= 0:
                print("\nMESSAGE DIED LOL\n")
                return
            # Check if message has reached destination
            if self.name == m.dest:
                debug(RX_DEBUG, "RX: Destination reached.")
                # print(m.data, " REACHED ", m.ttl)
                pass
            # Move from rx to tx
            else:
                if len(self.txBuffer) < self.maxBufferSize:
                    debug(RX_DEBUG, "RX: Moving message from RX to TX.")
                    self.txBuffer.append(m)
                    self.consume_power()
                else:
                    debug(RX_DEBUG, "RX: TX buffer is full.")
            m.path.append(self.name)
    
    # Message transit update
    def transmit(self):
        TX_DEBUG = 1
        # If node has no power, it is dead
        if self.power <= 0:
            debug(TX_DEBUG, "TX: Node %s is dead. Cannot transmit." % (self.name))
            return
        # Check node's TX to see if there is a message to be sent
        debug(TX_DEBUG, "TX: Processing %s's TX..." % (self.name))
        if len(self.txBuffer):
            debug(TX_DEBUG, "TX: \tTransmitting from %s..." % (self.name))
            # Peek at the message off the buffer
            # m = self.txBuffer[0]
            m = self.txBuffer.pop(0)
            # Acquire the next hop, function returns a Connection object
            nextHop = self._get_next_hop(m.path[-2] if len(m.path) >= 2 else m.path[-1]) 
            if nextHop is not None:
                debug(TX_DEBUG, "TX: \t\tLink (" + self.name + "->" + nextHop.node.name \
                     + ") with " + str(nextHop.reliability) + " reliability...")
                # Transmit message based on reliability (could be moved to another function at some point.)
                if random.random() <= nextHop.reliability:
                    debug(TX_DEBUG, "TX: \t\t\t...SUCCESS.")
                    # If connection is reliable then pop from buffer
                    # m = self.txBuffer.pop(0)
                    # Decrease ttl
                    m.ttl = m.ttl - nextHop.distance
                    transmitSuccess = nextHop.node.receive(m)
                    if transmitSuccess:
                        debug(TX_DEBUG, "TX: \t\tMessage sent.")
                    else:
                        debug(TX_DEBUG, "TX: \t\tMessage still failed (error on receiving end).")
                else:
                    debug(TX_DEBUG, "TX: \t\t\t...FAILED.")
                    transmitSuccess = False
                debug(TX_DEBUG, "TX: \tMessage from " + self.name + " to " + nextHop.node.name + " : " + str(transmitSuccess))
                # Record interaction for rewards
                self.record_reward(nextHop.node.name, transmitSuccess)
                # Deplete power, whether or not link is reached
                self.consume_power()
            else:
                debug(TX_DEBUG, "TX: \tNext-hop not available.")
    
    # Store into message buffer
    def receive(self, message):
        RX_DEBUG = 1
        # If node has no power, it is dead
        if self.power <= 0:
            debug(RX_DEBUG, "UP: Node %s is dead. Cannot receive." % (self.name))
            return False
        # If the buffer isn't full, store the message
        if len(self.rxBuffer) < self.maxBufferSize:
            self.rxBuffer.append(message)
            # debug(RX_DEBUG, "\tRX: Path after: %s" % (str(self.rxBuffer[-1].path)))
            # Deplete power
            self.consume_power()
            return True
        # If the buffer is full, return error
        else:
            return False

    # Send a message originating at this node
    def send(self, data, dest, ttl):
        m = Message(self.name, dest, data, ttl=ttl)
        if len(self.txBuffer) < self.maxBufferSize:
            self.txBuffer.append(m)
            # Deplete power
            self.consume_power()
    
    # Get congestion value (NOT COMPLETE)
    def get_congestion(self):
         return (float(len(self.rxBuffer)) + float(len(self.txBuffer))) \
              / (2 * self.maxBufferSize)

    # Power depletion function
    def consume_power(self):
        self.power = max(0, self.power - self.powerDeplete)

    # Record history
    def record_reward(self, nodeName, success):
        self.histories[nodeName].pop(0)
        self.histories[nodeName].append(1 if success is True else 0)
    
    # Get the learned reward based on the current experience history
    def get_record(self, nodeName):
        GR_DEBUG = 1
        # debug(GR_DEBUG, str(self.histories[nodeName]))
        past_rewards = []
        for i in range(len(self.histories[nodeName])):
            past_reward = ALPHA_RATE**i * self.histories[nodeName][-i]
            past_rewards.append(past_reward)
        learned_reward = sum(past_rewards) / MAX_LEARNED_REWARD
        return learned_reward

    # Construct neighbors list
    def add_neighbors(self, adjDict, nodeLookup):
        self.nodeLookup = nodeLookup
        # For each node in this node's entry in the adjacency list
        for n in adjDict[self.name]:
            # Append a neighbor object
            self.neighbors.append(Connection(self, self.nodeLookup[n[0]], n[1]))
            self.histories[n[0]] = [1] * HISTORY_WINDOW
            if LEARNING:
                self.Q[n[0]] = 2 if n[0] == self.sinkName else 0
                debug(0, "Node %s: Q['%s']" % (self.name, n))
    def costFunction(self,pr):
      cost=0;
      for sr in NODE_LIST:
        tem=sr
        vis=set({})
        curr_path_cost=0
        distance_costs_per_node=[]
        #print("starting from %d"%tem)
        while (True):
          if(tem==SINK):
            cost+=curr_path_cost
            distance_costs_per_node.append(curr_path_cost)
          if(tem==""):
            cost+=infinity
            break
          max=("",-1,infinity);#(node,priority,weight)
          vis.add(tem)
          for x in EDGE_DICT[sr]:
            if (x[0]==SINK):
              place=sink
            else:
              place=ord(x[0])-65
            if pr[place]>max[1] and x[0] not in vis:
              max=(x[0],pr[place],x[1])
          tem=max[0]
          curr_path_cost+=max[2]
      return cost

# Get distance to sink (Djikstra's, unoptimized)
    def get_shortest_dist_PSO(self,SINK, EDGE_DICT):
      alh = pso(30,self.costFunction,0,10,NUMBER_OF_NODE,50)
      best = alh.get_Gbest()
      pr=best
      distance_costs_per_node=[]
      for sr in NODE_LIST:
        tem=sr
        vis=set({})
        curr_path_cost=0
        while (True):
          if(tem==SINK):
            distance_costs_per_node.append(curr_path_cost)
          if(tem==""):
            break
          max=("",-1,infinity);#(node,priority,weight)
          vis.add(tem)
          for x in EDGE_DICT[sr]:
            if (x[0]==SINK):
              place=sink
            else:
              place=ord(x[0])-65
            if pr[place]>max[1] and x[0] not in vis:
              max=(x[0],pr[place],x[1])
          tem=max[0]
          curr_path_cost+=max[2]
      return min(distance_costs_per_node)
    
    def calculate_least_adjacent_cost(adjacency_list, i, v, cache):
        adjacent_nodes = adjacency_list[v]
        least_adjacent_cost = float("inf")
        for node in adjacent_nodes:
          adjacent_cost = cache[node["from"]-1] + node["weight"]
          if adjacent_cost < least_adjacent_cost:
            least_adjacent_cost = adjacent_cost
        return least_adjacent_cost
    
    def get_shortest_dist_DSDV(self,src,sink,adjacency_list):           
        cache = [[] for j in range(vertices)]
        cache[src] = 0
        for v in range(0, vertices):
          if v != src:
            cache[v] = float("inf")
            
        for i in range(1, vertices):
          print(cache)
          for v in range(0, vertices):
            previous_cache = cache
            least_adjacent_cost = calculate_least_adjacent_cost(adjacency_list, i, v, previous_cache)
            cache[v] = min(previous_cache[v], least_adjacent_cost)


        # detecting negative cycles
        for v in range(0, vertices):
          previous_cache = copy.deepcopy(cache)
          least_adjacent_cost = calculate_least_adjacent_cost(adjacency_list, i, v, previous_cache)
          cache[v] = min(previous_cache[v], least_adjacent_cost)

        if(not cache == previous_cache):
            raise Exception("negative cycle detected")
            
        shortest_path = min(cache)


    def get_shortest_dist_AODV(self, sinkName, adjDict, prevHop=[]):
        SD_DEBUG = 0
        debug(SD_DEBUG, "Getting shortest route for %s given %s..." % (self.name, str(prevHop)))
        # If sink is reached, return 0
        if self.name == sinkName:
            self.distance = 0
            return self.distance
        # Create list of viable neighbors
        viableNeighbors = []
        debug(SD_DEBUG, "%s\tAll neighbors: %s" % (self.name, str([n.node.name for n in self.neighbors])))
        debug(SD_DEBUG, "%s\t\tAll distances: %s" % (self.name, str([n.distance for n in self.neighbors])))
        for neighbor in self.neighbors:
            if (neighbor.node.name not in prevHop):
                viableNeighbors.append(neighbor)
        debug(SD_DEBUG, "%s\t\tViable neighbors: %s" % (self.name, str([n.node.name for n in viableNeighbors])))
        # If there's a dead end, return infinity
        if len(viableNeighbors) == 0:
            return INFINITY
        # Get possible distances
        possibleDists = []
        for n in viableNeighbors:
            distToNeighbor = n.distance
            debug(SD_DEBUG, "%s\tDist. to %s: %d" % (self.name, n.node.name, n.distance))
            possibleDists.append(distToNeighbor + \
                n.node.get_shortest_dist(sinkName, adjDict, prevHop=prevHop+[self.name]))
        debug(SD_DEBUG, "%s\tPossible dists. given %s: %s" % (self.name, str(prevHop), str(possibleDists)))
        # If we're back at the top level of recursion, assign the distance
        if len(prevHop) == 0:
            self.distance = min(possibleDists + [self.distance])
            debug(SD_DEBUG, "%s\t\tAssigning own dist. to sink: %d" % (self.name, self.distance))
            return self.distance
        # If we're still in the recursion, just return the min of possible distances
        return min(possibleDists)
    
    # Get next hop based on Q-learning or naive approach
    def _get_next_hop(self, prevHop):
        NH_DEBUG = 1
        debug(NH_DEBUG, "NH: Getting next hop for %s" % (self.name))
        # Create list of viable neighbors
        viableNeighbors = []
        debug(NH_DEBUG, "NH: \tPrevious hop: %s" % (prevHop))
        for neighbor in self.neighbors:
            # debug(NH_DEBUG, "NH: \t\tChecking %s" % (neighbor.node.name))
            if (neighbor.node.name is not prevHop):
                viableNeighbors.append(neighbor)
        debug(NH_DEBUG, "NH: \tViable neighbors: %s" % (str([n.node.name for n in viableNeighbors])))
        # Return neighbor based on best inherent Q-value
        if len(viableNeighbors):
            debug(NH_DEBUG, "NH: \tQ-values: %s" % (str(self.Q)))
            if LEARNING:
                nextHopOptions = (sorted(viableNeighbors, key=(lambda x: self.Q[x.node.name])))
            else:
                nextHopOptions = (sorted(viableNeighbors, key=(lambda x: x.node.Q)))
            for pn in nextHopOptions:
                if LEARNING:
                    debug(NH_DEBUG, "NH:\t\t%s: %f %f" % (pn.node.name, self.Q[pn.node.name], self.get_record(pn.node.name)))
                else:
                    debug(NH_DEBUG, "NH:\t\t%s: %f %f" % (pn.node.name, pn.node.Q, self.get_record(pn.node.name)))
            nextHop = nextHopOptions[-1]
            debug(NH_DEBUG, "NH: \t\tNext hop: %s" % (nextHop.node.name))
            return nextHop
        else:
            return None
    
    # Naive Deterministic Approach (SAdaR-D)
    def generate_Q_value_naive(self, depthLevel=0):
        if self.name == self.sinkName:
            return 2
        r = W_D * (1 - float(self.distance) / self.maxDistance) + \
            W_L * (self.reliability) + \
            W_C * (1 - self.congestion) + \
            W_P * (self.power)
        if depthLevel < Q_DEPTH:
            return r + DISCOUNT_RATE * max([n.node.generate_Q_value_naive(depthLevel=depthLevel+1) for n in self.neighbors])
        else:
            return r
    
    # Simple Decision Approach and Q-Learning Approach (SAdaR-SD: zeta = 1.0, SAdaR-RL: zeta < 1.0)
    def generate_Q_value_plus(self, depthLevel=0):
        GQ_DEBUG = 0
        # If we're dealing with the sink, assign arbitrary value higher than max achievable Q-value
        if self.name == self.sinkName:
            return {self.name: 4}
        Q = {}
        # For each neighbor, calculate...
        for n in self.neighbors:
            if n.node.name == self.sinkName:
                Q[n.node.name] = 2
            # Inherent perceived value
            inherent_value = W_D * (1 - float(n.node.distance) / self.maxDistance) + \
                W_L * (n.reliability) + \
                W_C * (1 - n.node.congestion) + \
                W_P * (n.node.power)
            # Learned reward value
            learned_reward = self.get_record(n.node.name)
            if depthLevel < Q_DEPTH:
                debug(GQ_DEBUG, "%s->%s" % (self.name, n.node.name))
                max_expected_reward = max(n.node.generate_Q_value_plus(depthLevel=depthLevel+1).values())
                # Maximum Expected Future Reward
                Q_val = (1 - BETA_RATE) * inherent_value + BETA_RATE * (learned_reward + DISCOUNT_RATE * max_expected_reward)
            else:
                Q_val = (1 - BETA_RATE) * inherent_value + BETA_RATE * learned_reward
            Q[n.node.name] = (1 - ZETA_RATE) * self.Q[n.node.name] + ZETA_RATE * Q_val
        return Q
