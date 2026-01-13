import json
from parser import Parser
from commands import Command
from optimizer import Optimizer

def main():
    file_path = "./problem.json"
    with open(file_path, "r") as f:
        problem = json.load(f)

    dsl = [line.strip() for line in problem["answer"].split('\n') if line.strip()]
    print(dsl)

    parse = Parser()
    print(parse.parse_sexprs(dsl))

    command_reader = Command(dsl)
    points = command_reader.points
    for p in points:
        print(f"Command points: {p}")

    opts = {'epochs': 1000, 'learning_rate': 0.01}
    optimizer = Optimizer(
        command_reader.instructions,
        opts,
        verbosity=True
    )

    diagram = optimizer.solve()

    diagram.plot(show=True)


if __name__ == "__main__":
    main()