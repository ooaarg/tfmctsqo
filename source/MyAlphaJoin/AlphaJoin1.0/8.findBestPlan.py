from __future__ import division
import time
import math
import random
from copy import deepcopy
import numpy as np
import torch
import os

from models import ValueNet

model_path = "./saved_models/supervised.pt"
shortToLongPath = "../resource/shorttolong"
predicatesEncodeDictPath = "./predicatesEncodedDict"

# Load dictionaries to reconstruct the actual input dimensionality,
# as in the supervised class (len(tables)^2 + len(predicatesEncodeDict['1a'])).
if not os.path.exists(shortToLongPath) or not os.path.exists(predicatesEncodeDictPath):
    raise RuntimeError(
        "shorttolong or predicatesEncodedDict not found; run 2.getQueryEncode.py first."
    )

with open(shortToLongPath, "r", encoding="utf-8") as f:
    short_to_long = eval(f.read())

tables = sorted(short_to_long.keys())

with open(predicatesEncodeDictPath, "r", encoding="utf-8") as f:
    predicatesEncodeDict = eval(f.read())

any_key = next(iter(predicatesEncodeDict.keys()))
pred_dim = len(predicatesEncodeDict[any_key])
num_inputs = len(tables) * len(tables) + pred_dim
# For the current training setup this is effectively ValueNet(856, 5),
# but computing num_inputs from metadata keeps it correct if schemas change.
predictionNet = ValueNet(num_inputs, 5)
predictionNet.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage)
)
predictionNet.eval()


def getReward(state):
    inputState = torch.tensor(state.board + state.predicatesEncode, dtype=torch.float32)
    with torch.no_grad():
        predictionRuntime = predictionNet(inputState)
    prediction = predictionRuntime.detach().cpu().numpy()
    maxindex = np.argmax(prediction)
    reward = (4 - maxindex) / 4.0
    return reward


def randomPolicy(state):
    """
    Random rollout until a terminal state.
    If for some reason the state is marked as non-terminal,
    but there are no available actions (empty list), we treat it as terminal
    to avoid crashing with an error.
    """
    while True:
        if state.isTerminal():
            return getReward(state)
        actions = state.getPossibleActions()
        if not actions:
            # Non-terminal state with no actions — stop the search and evaluate as is.
            return getReward(state)
        action = random.choice(actions)
        state = state.takeAction(action)


class treeNode:
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class mcts:
    def __init__(
        self,
        iterationLimit=None,
        explorationConstant=1 / math.sqrt(2),
        rolloutPolicy=randomPolicy,
    ):
        if iterationLimit == None:
            raise ValueError("Must have either a time limit or an iteration limit")
        # number of iterations of the search
        if iterationLimit < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.searchLimit = iterationLimit
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)
        for i in range(self.searchLimit):
            self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        # If the root has no children (no move could be made at all),
        # return None so the caller can skip this query.
        if bestChild is self.root:
            return None
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        newState = deepcopy(node.state)
        reward = self.rollout(newState)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        # If there are no actions but the node is marked as non-terminal,
        # treat it as effectively terminal and mark it as fully expanded.
        if not actions:
            node.isFullyExpanded = True
            node.isTerminal = True
            return node
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                # if newNode.isTerminal:
                #     print(newNode)
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        # If the node has no children, return the node itself (for the root this is a signal to the caller).
        if not node.children:
            return node

        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = (
                child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            )
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
