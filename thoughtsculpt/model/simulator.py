from collections import defaultdict 
import numpy as np

class MCTS:
    def __init__(self, exploration_weight=1, exact=True, num_candidates=3):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.nodes= set()
        self.exploration_weight = exploration_weight
        self.exact = exact
        self.num_candidates = num_candidates

    def find_node(self, node, num_iter=3, depth=1):
        for d in range(depth):
            for i in range(num_iter):
                self.do_rollout(node)
            node = self.choose(node)
        return node


    def choose(self, node=None):
        "Choose the best node."

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves and moves that only explored once
            return self.Q[n] / (1 + self.N[n])  # average reward
        
        if node is not None:
            if node.is_terminal():
                return max(self.nodes, key=lambda x: x.reward())
            else:
                return max(self.children[node], key=score)
        return max(self.nodes, key=lambda x: x.reward())

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf, num_candidates=self.num_candidates)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = [node]
        while True:
            if node not in self.children or not self.children[node] or node.is_terminal():
                # node is either unexplored or terminal
                return path
            else:
                node = self._uct_select(node)  # descend a layer deeper
                path.append(node)


    def _expand(self, node, num_candidates=1):
        "Update the `children` dict with the children of `node`"
        self.nodes.add(node)
        if node in self.children and len(self.children[node]) > 0:
            return  # already expanded
        elif node.is_terminal():
            self.children[node] = []
            return
        candidates = node.get_candidates(num_candidates=num_candidates)
        candidates = [n for n in candidates if n not in self.nodes]
        if len(candidates) == 0:
            return self._expand(node, num_candidates=num_candidates+1)
        else:
            self.children[node] = candidates
        
        self.nodes.update(self.children[node])
        for child in self.children[node]:
            self.N[child] = 1
            self.Q[child] = child.reward()
    
    def _simulate(self, node, simulation_depth=1):
        "Returns the reward for a random simulation (to completion) of `node`"
        if node.is_terminal():
            return node.reward()
        if node in self.children and len(self.children[node]) > 0:
            node = np.random.choice(self.children[node])
        if simulation_depth == 1:
            reward = node.reward()
            return reward
        else:
            node = node.get_candidates(num_candidates=1)[0]
            return self._simulate(node, simulation_depth=simulation_depth-1)
            
        

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        # All children of node should already be expanded:

        log_N_vertex = np.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            
            
            return self.Q[n]  + self.exploration_weight * np.sqrt(
                log_N_vertex / (1 + self.N[n])
            )

        return max(self.children[node], key=uct)

class SelfRefine:
    def __init__(self):
        self.nodes = []
    
    def choose(self):
        return max(self.nodes, key=lambda x: x.reward())

    def find_node(self, node, depth=1, num_iter=1):
        for i in range(depth):
            node = node.get_candidates(num_candidates=1)[0]
            self.nodes.append(node)
        return node

  
class DFS:
    def __init__(self):
        self.explored_nodes = set()
        self.children = {}     

    def choose(self):
        return max(self.explored_nodes, key=lambda x: x.reward())

    def find_node(self, node, depth=1, num_iter=1):
        self.explored_nodes.add(node)
        for i in range(depth):

            if node.is_terminal():
                if node.can_end():
                    return node
                node = node.parent
            elif node not in self.children:
                self.children[node] = node.get_candidates()
            candidates = [n for n in self.children[node] if n not in self.explored_nodes]
            if len(candidates) == 0:
                self.children[node] = node.get_candidates()
                
            else:
                node = max(candidates, key=lambda x: x.reward())
            self.explored_nodes.add(node)
        return node

class ToT:
    def __init__(self):
        self.explored_nodes = set()
        self.children = {}
        

    def choose(self):
        return max(self.explored_nodes, key=lambda x: x.reward())

    def find_node(self, node, depth=1, num_iter=1):
        self.explored_nodes.add(node)
        for i in range(depth):
            if node.is_terminal():
                # if node.can_end():
                return node
                # node = node.parent
            elif node not in self.children:
                self.children[node] = node.get_tot_candidates()
            candidates = [n for n in self.children[node] if n not in self.explored_nodes]
            if len(candidates) == 0:
                self.children[node] = node.get_tot_candidates()
                
            else:
                node = max(candidates, key=lambda x: x.reward())
            self.explored_nodes.add(node)
        return node


class COT:
    def __init__(self):
        pass
    def find_node(self, node):
        return node.get_cot_candidate().position



