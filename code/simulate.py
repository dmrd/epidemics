#!./ENV/bin/pypy
# encoding: utf-8
#
### Simulation framework for COS445 final project on choosing influential nodes
### in dynamic graphs
#
# David Dohan
# David Durst
#
#
# The following collaboration network is probably best for testing:
# http://snap.stanford.edu/data/ca-GrQc.html
#
# This one (or similar) is used in the original paper (also good to test on)
# http://snap.stanford.edu/data/ca-HepTh.html
import math
import sys
import pickle
import random
import time
import decimal
import networkx as nx

#Number monte carlo trials to estimate activation for each node in greedy
NUMGREEDYTRIALS = 100
#Number of monte carlo trials for final activation estimate
NUMOUTPUTTRIALS = 1000
#Amount of change each round for each edge
DELTA = 0.1
#How far to move in sin curve each time step
SINDELTA = math.pi/2
#Have we seeded the random number generator yet?
SEEDED = False

#initialize the edge 
def edgeInit(edgeObj):
    edgeObj['probability'] = random.uniform(0, 1)
    edgeObj['direction'] = -2 * random.randint(0, 1) + 1
    edgeObj['lastUpdateTime'] = 0
    edgeObj['sinStart'] = random.uniform(0, 2*math.pi)

#What is the current probability for the edge at the given time?
def infectProb(edge, time, G):
    edgeObj = G[edge[0]][edge[1]]
    #three different options for calculating probability change over time
    if (len(sys.argv) < 5 or (sys.argv[4] != 'linear' and sys.argv[4] != 'sine')): 
        edgeObj['probability'] = math.pow(DELTA, edgeObj['direction'] * \
            (time - edgeObj['lastUpdateTime'])) * edgeObj['probability']
    elif (sys.argv[4] == 'linear'):
        edgeObj['probability'] = DELTA * edgeObj['direction'] * \
            (time - edgeObj['lastUpdateTime']) + edgeObj['probability']
    else:
        edgeObj['probability'] = math.sin(edgeObj['sinStart'] + time*SINDELTA)
    #fix up so that probabilties don't go haywires too big or small
    if (edgeObj['probability'] > 1):
        edgeObj['probability'] = 1
    if (edgeObj['probability'] < 0):
        edgeObj['probability'] = 0
    return edgeObj['probability']

#def sinProbj

#Simulate the spread function
def monteCarloSpread(G, activeSet):
    allActive = set(activeSet)
    unchanged = False
    timeStep = 1
    while (not(unchanged)):
        unchanged = True
        nextSet = set()
        for node in activeSet:
            for nbr in G[node]:
                if nbr not in allActive:
                    if (random.random() < infectProb((node,nbr), timeStep, G)):
                        unchanged = False
                        nextSet.add(nbr)
                        allActive.add(nbr)
        timeStep += 1
        activeSet = nextSet
    return len(allActive)

def monteCarloTrials(G, startSet, trials=10000):
    total = 0.0
    for i in range(trials):
        #print(i)
        total += monteCarloSpread(G, startSet)
    return (total / trials)


def greedy(G, K):
    #Use a list so we can get the order they were added
    currentSet = []
    for i in range(K):
        timeStart = time.time()
        bestNode = None
        bestVal = 0
        for node in G.nodes():
            if node in currentSet:
                continue
            currentSet.append(node)
            active = monteCarloTrials(G, currentSet, NUMGREEDYTRIALS)
            currentSet.pop()
            if (active > bestVal):
                bestVal = active
                bestNode = node
        if (bestNode == None):
            print("No best node - error")
            return set()
        currentSet.append(bestNode)
        timeEnd = time.time()
        print("Node " + str(i+1) + ": " + str(bestVal))
        print("\tTime: " + str(timeEnd-timeStart))
        sys.stdout.flush()
    return currentSet
    
#Return the K nodes with the smallest average distance to other nodes in G
def centrality(G, K):
    #See if we have done this before.  It's an expensive operation.
    try:
        saved = open(sys.argv[1] + ".centrality", 'rb')
        print("Saved centrality results found")
        result = pickle.load(saved)
        saved.close()
        return set(result[:K])
    except Exception:
        print("No saved centrality results found")
        print("This might take a while...")
        #Sort nodes by average distance to others in their connected component
        degSorted = sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1])
        degSorted = [n[0] for n in degSorted[-1::-1]]
        #Save the result - expensive operation
        toSave = open(sys.argv[1] + ".centrality", 'wb')
        pickle.dump(degSorted,toSave,-1)
        toSave.close()
        #Return set of nodes - strip degree count (degSorted is list of tuples)
        return degSorted[:K]

#Return the K nodes with the highest degree in G
def degree(G, K):
    #Sort nodes by degree
    degSorted = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1])
    #Return set of nodes - strip degree count (degSorted is list of tuples)
    return [n[0] for n in degSorted[:-K-1:-1]]


#Returns a set of K randomly chosen nodes from G
def randomNodes(G, K):
    return set(random.sample(G.nodes(), K))

def simulate(G):
    K = int(sys.argv[3]) #Set size
    alg = sys.argv[2] #Heuristic
    #try to seed the if random number generator if the seed has been specified
    if (len(sys.argv) >= 6):
        random.seed(int(sys.argv[5]))
    #initialize all nodes
    for node in G.nodes():
        for nbr in G[node]:
            edgeObj = G[node][nbr]
            #initialize the edge if it doesn't have all the desired fields
            if ('lastUpdateTime' not in edgeObj):
                edgeInit(edgeObj)
    if (alg == "degree"):
        S = degree(G, K)
    elif (alg == "centrality"):
        S = centrality(G, K)
    elif (alg == "random"):
        S = randomNodes(G, K)
    elif (alg == "greedy"):
        S = greedy(G, K)
    else:
        print("No heuristic called " + sys.argv[3] + " found")
        return -1

    print("Estimating activation function...")
    print(S)
    print(monteCarloTrials(G, S, NUMOUTPUTTRIALS))


def main():
    if (len(sys.argv) < 4):
        print("syntax:" + str(sys.argv[0]) + " GraphFile Heuristic SetSize ProbabilityAlgorithm Seed")
        print("Heuristics: degree, centrality, greedy, and random")
        print("ProbabilityAlgorithm: tail, linear, and sine")
        return
    else:
        print("Reading " + str(sys.argv[1]))
        G = nx.read_edgelist(sys.argv[1],'#',create_using=nx.MultiGraph())
        print("Running simulation with:")
        print("Set size: \t" + sys.argv[3])
        print("Heuristic: \t" + sys.argv[2])
        print("Nodes: \t\t" + str(G.order()))
        print("Edges: \t\t" + str(len(G.edges())))
        simulate(G)

if __name__ == '__main__':
    random.seed()
    main()
