from parse import Parser
import json
from command import Command
from optimizer import Optimizer

# text = "Cho đường tròn tâm O đi qua điểm A"
# sexprs = [
#     "(param A point)",
#     "(param O point)",
#     "(param omega_O circle)",
#     "(define omega_O circle (coa O A))"
# ]

def main():
    file_path = "./problem.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        problem = json.load(f)
        
    dsl = [line.strip() for line in problem["answer"].split('\n') if line.strip()]
    print(dsl)    
    
    parse = Parser()
    print(parse.parse_sexprs(dsl))
    
    command = Command(dsl)
    points = command.points
    for p in points:
        print(f"command point: {p}")
    
    opts = {'epochs': 1000, 'learning_rate': 0.01}
    optimizer = Optimizer(
        command.instructions,
        opts,
        verbose=True
    )

    diagram = optimizer.solve()
    diagram.plot(show=True)
    
if __name__ == "__main__":
    main()
     
        


