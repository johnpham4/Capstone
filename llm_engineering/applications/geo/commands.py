from .parser import Parser

class Command:
    def __init__(self, problem_lines: list[str]):
        self.points = list()
        self.lines = list()

        self.instructions = list()
        self.problem_lines = problem_lines

        self.unnamed_points = list()
        self.unnamed_lines = list()
        self.segments = list()
        self.seg_colors = list()

        cmds = Parser.parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except:
                raise RuntimeError(f"Invalid command: {cmd}")

    @classmethod
    def process_command(cls, cmd: list):
        if not isinstance(cmd[0], str):
            raise RuntimeError(f"[process_cmd] command must be a string")
        head = cmd[0].lower()
        if head == "assert":
            cls.add(cmd)
        elif head == "triangle":
            pass
        elif head == "eval":
            cls.eval_cons(cmd)
        else:
            raise NotImplementedError(f"[Command.process_command] Command not supported: {head}")

    def add(self, cmd: list):
        assert (len(cmd) == 2)
