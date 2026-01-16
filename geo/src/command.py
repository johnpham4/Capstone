from parse import Parser
from primitives import *
from instructions import parameter

# Format: (param (A B C) (iso-tri A))

class Command:
    def __init__(self, problem_lines: list[str]):
        self.points = list()
        self.lines = list()
        
        self.instructions = list()
        self.problem_lines = problem_lines    
        
        cmds = Parser.parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except Exception as e:  
                print(f"Error processing command {cmd}: {e}")
    
    def register_point(self, point):
        """Register a point if not already registered"""
        if point not in self.points:
            self.points.append(point)
    
    def process_command(self, cmd):
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must start with a string")
        
        head = cmd[0]
        if head == "param":
            self.process_param(cmd)
        elif head == "assert":
            self.process_assert(cmd)
        else:
            raise NotImplementedError(f"Command not supported: {head}")
        
    def process_param(self, cmd):
        if isinstance(cmd[1], tuple): # multiple objects like (param (A B C) (iso-tri A))
            ps = [Point(p) for p in cmd[1]]
            for p in ps:
                self.register_point(p)
                
            param_method = cmd[2]
            if isinstance(param_method,str): # no arguments
            # cmd = ['param', ('A', 'B', 'C'), 'triangle']
                instr = parameter(ps, param_method)
                self.instructions.append(instr)
                
            elif isinstance(param_method, tuple): # with arguments like (iso-tri A)
            # cmd = ['param', ('A', 'B', 'C'), ('iso-tri', 'A')]
                method_name = param_method[0].lower()
                args = param_method[1]
                instr = parameter(ps, method_name, args)
                
                args_points = Point(args)
                instr = parameter(ps, method_name, (args_points,))
                self.instructions.append(instr)
        else:
            # single object like (param D point (on-seg A B))
            obj_name = cmd[1]
            obj_type = cmd[2]
            
            if obj_type == "point":
                p = Point(obj_name)
                self.register_point(p)
                
                if len(cmd) > 3:
                    constraint = cmd[3]
                    if isinstance(constraint, tuple):
                        constr_name = cmd[3]
                        if isinstance(constr_name, str):
                            predicate  = constr_name.lower()
                            args = [Point(a)for a in constraint[1:]]
                            instr = parameter([p],predicate, args)
                            self.instructions.append(instr)
            
         
        