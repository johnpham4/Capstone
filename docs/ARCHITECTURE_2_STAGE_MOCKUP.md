# GeoUni - 2-Stage Architecture (Mockup)

## Mermaid Diagram
```mermaid
flowchart LR
    A[Input\nDe hinh hoc tieng Viet] --> B

    subgraph S1[Stage 1: Neural DSL Generation]
      direction LR
      B[Fine-tuned LLM\n(AceMath/Qwen family)] --> C[GMBL/DSL Generator]
      C --> D[Structured DSL Output\nS-expression]
    end

    D --> E[S-Expression Parser\nDSLParser]
    E --> F[Constraint Builder\nDiagramBuilder\nParameter + Assertion]

    subgraph S2[Stage 2: Constraint-based Numeric Optimization]
      direction TB
      F --> G[Smart Initializer\ntriangle/quadrilateral/circle priors]
      G --> H[Loss Construction\nL_parallel, L_perp, L_dist, L_angle, L_on_circle, L_NDG, ...]
      H --> I[Adam Optimizer\neta=0.01, epochs, n_tries]
      I --> J[Post-correction\nincircle/tangent consistency]
    end

    J --> K[Matplotlib Renderer]
    K --> L[Diagram Output\nPNG/base64]
```

## ASCII Fallback (neu Mermaid khong hien)
```text
Input (de hinh hoc tieng Viet)
    |
    v
[Stage 1: Neural DSL Generation]
    Fine-tuned LLM (AceMath/Qwen family)
            -> GMBL/DSL Generator
            -> Structured DSL (S-expression)
    |
    v
S-Expression Parser (DSLParser)
    -> Constraint Builder (DiagramBuilder)

[Stage 2: Constraint-based Numeric Optimization]
    Smart Initializer
        -> Loss Construction
            (L_parallel, L_perp, L_dist, L_angle, L_on_circle, L_NDG, ...)
        -> Adam Optimizer (eta=0.01, epochs, n_tries)
        -> Post-correction (incircle/tangent)
    |
    v
Matplotlib Renderer
    -> Diagram Output (PNG/base64)
```
