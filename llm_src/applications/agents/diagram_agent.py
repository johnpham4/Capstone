"""Diagram Renderer Agent - converts DSL to visual diagram."""

from typing import Dict, Any
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger

from llm_src.domains.orchestration import Agent, AgentType, AgentState
from llm_src.applications.diagram.diagram_builder import DiagramBuilder
from llm_src.applications.diagram.optimizer import Optimizer
from llm_src.infrastructures.visualization.matplotlib_renderer import MatplotlibDiagramRenderer


class DiagramRendererAgent(Agent):
    """
    Render diagram from DSL code.

    Steps:
    1. Parse DSL → DiagramBuilder
    2. Optimize layout → Optimizer
    3. Render to PNG → MatplotlibRenderer
    """

    def __init__(self, epochs: int = 2000, dpi: int = 150):
        super().__init__(AgentType.DIAGRAM_RENDERER)
        self.epochs = epochs
        self.dpi = dpi

    async def execute(self, state: AgentState) -> AgentState:
        """Render diagram from DSL."""
        state.add_execution_step(self.name)

        if not state.dsl:
            error_msg = "No DSL available for rendering"
            state.add_error(error_msg)
            state.diagram_error = error_msg
            logger.warning(error_msg)
            return state

        try:
            # Parse DSL
            dsl_lines = state.dsl.split('\n') if '\n' in state.dsl else [state.dsl]
            dsl_lines = [line.strip() for line in dsl_lines if line.strip()]

            logger.info(f"Parsing {len(dsl_lines)} DSL lines")
            builder = DiagramBuilder(dsl_lines)

            # Optimize
            opts = {
                'epochs': self.epochs,
                'n_tries': 1,
                'eps': 1e-6,
                'seed': 42
            }
            logger.info(f"Optimizing diagram (epochs={self.epochs})")
            optimizer = Optimizer(builder.instructions, opts, verbosity=False)
            diagram = optimizer.solve()

            # Render
            logger.info("Rendering diagram")
            renderer = MatplotlibDiagramRenderer()
            fig, ax = renderer.render(diagram, save=False, show=False)

            # Add title from user input
            if state.user_input:
                title = state.user_input[:100]  # Truncate if too long
                fig.suptitle(title, fontsize=10, wrap=True)

            # Convert to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            state.diagram_bytes = buf.getvalue()
            plt.close(fig)

            logger.info(f"Diagram rendered ({len(state.diagram_bytes)} bytes)")

            # Store diagram info in context
            state.context['diagram_info'] = {
                'num_points': len(builder.points),
                'num_lines': len(dsl_lines),
                'size_bytes': len(state.diagram_bytes)
            }

        except Exception as e:
            error_msg = f"Diagram rendering failed: {str(e)}"
            state.add_error(error_msg)
            state.diagram_error = error_msg
            logger.error(error_msg, exc_info=True)

        return state

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "type": self.agent_type.value,
            "epochs": self.epochs,
            "dpi": self.dpi
        }
