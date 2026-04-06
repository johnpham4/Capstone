from src.services.orchestration.agents.solver_agent import SolverAgent

problem = "Cho hình thang cân ABCD có AB CD , đường chéo DB vuông góc với cạnh bên BC , DB là tia phân giác góc D . Tính chu vi của hình thang, biết 3BC  cm."


def main() -> None:
	agent = SolverAgent()
	for chunk in agent.stream_solve(user_input=problem):
		print(chunk, end="", flush=True)
	print()


if __name__ == "__main__":
	main()
