
from thoughtsculpt.model.node import ContentNode, CrosswordNode, CommonNode
from thoughtsculpt.model.node24 import Node24
from thoughtsculpt.model.simulator import MCTS, COT

class ContentImprover:
    def __init__(self, model, evaluator, solver_class=MCTS):

        self.model = model
        self.evaluator = evaluator
        self.solver_class = solver_class
    
    def improve(self, outlines, depth=3, continous=False):
        node = ContentNode(model=self.model, 
                           parent=None,
                           position=outlines,
                           )
        solver = self.solver_class()
        best_positions = []
        if self.solver_class == MCTS:
            original_node = node
            for d in range(depth):
                if self.evaluator is not None:
                    score = self.evaluator.evaluate_score(node.position)
                    print("Score:", score)
                node = solver.find_node(original_node, depth=1)
                best_positions.append(solver.choose().position)
                if node.is_terminal():
                    break
            node = solver.choose()
        
        elif self.solver_class == COT:
            return solver.find_node(node)

            
        else:
            for d in range(depth):
                if self.evaluator is not None:
                    score = self.evaluator.evaluate_score(node.position)
                    print("Score:", score)
                node = solver.find_node(node)
                best_positions.append(solver.choose().position)
                if node.is_terminal():
                    break

            node = solver.choose()
        if continous:
            return best_positions
        return node.position

class CrosswordImprover:
    def __init__(self, model, evaluator, solver_class=MCTS):

        self.model = model
        self.evaluator = evaluator
        self.solver_class = solver_class
    
    def improve(self, env, depth=3, scores={"r_word":[], "r_letter":[]}):
        
        node = CrosswordNode(model=self.model, 
                           parent=None,
                           env=env,
                           )
        solver = self.solver_class()
        r_word, r_letter = 0, 0
        for d in range(depth):
            node = solver.find_node(node, num_iter=3, depth=1)
            score = self.evaluator.evaluate_score(node.position)
            print("Score:", score)
            print(node)
            r_word = max(r_word, score["r_word"])
            r_letter = max(r_letter, score["r_letter"])
            if score["r_word"] == 1:
                break

        scores["r_word"].append(r_word)
        scores["r_letter"].append(r_letter)

        node = solver.choose()
        return node.position

class CommonGenImprover:
    def __init__(self, model, solver_class=MCTS, **kwargs):

        self.model = model
        self.solver_class = solver_class
        self.use_feedback = kwargs.get("use_feedback", True)
    
    def improve(self, concepts, scores={"base":[], "improved":[], "origs":[], "news":[]}, depth=3, verbose=False):
        node = CommonNode(model=self.model, concepts=concepts, use_feedback=self.use_feedback)
        solver = self.solver_class()
        best_score = node.get_future_score()
        scores["base"].append(best_score)
        scores["origs"].append(node.position)
        for d in range(depth):
            node = solver.find_node(node)
            score = node.get_future_score()
            best_score = max(score, best_score)
            print("Score:", score)
            if verbose:
                print(node)
            if score == 1:
                break
        node = solver.choose()
        scores["improved"].append(best_score)
        scores["news"].append(node.position)
        return node.position
            

class Game24Improver:
    def __init__(self, model, solver_class=MCTS):
        self.model = model
        self.solver_class = solver_class

    def improve(self, instruction, depth=3):
        node = Node24(instruction=instruction, model=self.model)
        solver = self.solver_class(num_candidates=5)
        best_positions = []
        for d in range(depth):
            solver.find_node(node)
            best_positions.append(solver.choose().position)
            if node.is_terminal():
                # print(node.position, "fb", node.feedback)
                break
        node = solver.choose()
        return node.position, solver
            
            
        
        
        
