# **I. PROJECT INTRODUCTION**

## **1. OVERVIEW**

### **1.1 Project Information**

**Project Title:** GeoUni - An AI-Powered Platform for Automated Geometry Diagram Generation and Problem Solving

**Project Code:** AIP491_CP

**Group Name:** [Tên nhóm]

**Academic Year:** 2025-2026

**Project Team:**
- [Họ và tên 1] - [Student ID] - Team Leader & Backend Developer
- [Họ và tên 2] - [Student ID] - Machine Learning Engineer
- [Họ và tên 3] - [Student ID] - Full-stack Developer

**Academic Supervisor:** [Họ và tên giảng viên]

**Institution:** FPT University, School of Information Technology

**Project Duration:** September 2025 - February 2026

### **1.2 Project Overview**

This report presents the development and implementation of GeoUni, an artificial intelligence-powered platform designed to address challenges in geometry education for Vietnamese high school students. The system provides two core functionalities: automated diagram generation from domain-specific language representations, and intelligent dataset processing for training machine learning models on geometry reasoning tasks.

The platform architecture follows a microservices design pattern, implemented using FastAPI framework for the backend API layer, PyTorch for constraint optimization algorithms, and AWS SageMaker for model deployment infrastructure. The system processes Vietnamese geometry problem statements, converts them into formal geometric representations using a custom domain-specific language (DSL), and generates accurate diagram visualizations through gradient-based constraint satisfaction techniques.

This document is organized into several major sections. Following this introduction, Section II provides technical background and related work analysis. Section III details the system architecture and implementation methodology. Section IV presents experimental results and performance evaluation. Section V discusses findings and implications, while Section VI concludes with contributions and future research directions.

---

## **2. PROJECT BACKGROUND**

The teaching and learning of geometry in Vietnamese secondary education faces several persistent challenges that impact student comprehension and performance. Traditional pedagogy relies heavily on static textbook diagrams and manual construction exercises, which often fail to provide students with intuitive understanding of spatial relationships and geometric properties. Furthermore, the availability of interactive learning tools for Vietnamese-language geometry instruction remains limited compared to resources available for other languages.

Recent advances in artificial intelligence, particularly in the domains of natural language processing and computer vision, present opportunities to develop intelligent tutoring systems for mathematics education. Large language models such as GPT-4 and Claude have demonstrated remarkable capabilities in mathematical reasoning tasks. Simultaneously, gradient-based optimization methods using deep learning frameworks like PyTorch have shown effectiveness in solving constraint satisfaction problems, including geometric configurations.

However, existing geometry software tools such as GeoGebra and Desmos, while powerful, require significant manual interaction and geometric knowledge to construct diagrams correctly. These systems lack automation capabilities for converting textual problem descriptions into accurate geometric visualizations. Additionally, there is a notable absence of comprehensive datasets containing Vietnamese geometry problems with formal representations suitable for machine learning applications.

The convergence of these educational needs and technological capabilities motivates the development of an integrated platform that can automatically interpret geometry problem descriptions in formal language, generate accurate diagrams satisfying multiple geometric constraints, and provide infrastructure for training AI models on geometry reasoning tasks. Such a system has potential to enhance geometry education accessibility and effectiveness for Vietnamese students while contributing to research in educational technology and AI-assisted learning.

---

## **3. PROJECT OBJECTIVE**

The primary objective of this project is to design, implement, and evaluate an artificial intelligence platform that supports Vietnamese high school students in learning plane geometry through automated diagram generation and AI-powered problem understanding.

### **3.1 General Objective**

To develop a production-ready platform that leverages constraint optimization and machine learning techniques to provide automated diagram generation and problem formalization capabilities, thereby improving learning outcomes and reducing barriers to geometry education for Vietnamese students.

### **3.2 Specific Objectives**

**Objective 1: Constraint-Based Diagram Generation System**

Develop a diagram generation module capable of converting domain-specific language representations into geometrically accurate visualizations. This includes: (a) designing a geometry DSL with sufficient expressiveness to represent common high school geometry problems; (b) implementing a parser that converts DSL specifications into constraint optimization problems; (c) developing a PyTorch-based optimization engine that solves geometric constraints through gradient descent; and (d) creating a rendering pipeline that produces clear, publication-quality diagram outputs.

**Objective 2: Dataset Processing and Model Training Pipeline**

Construct a comprehensive dataset processing system and establish a complete training pipeline for fine-tuning large language models on geometry tasks. This encompasses: (a) collecting and annotating geometry problems from Vietnamese educational sources; (b) developing automated pipelines to convert problem statements into formal DSL representations; (c) implementing a supervised fine-tuning methodology using instruction-following datasets; and (d) achieving validation accuracy exceeding 85% on held-out test problems for Vietnamese-to-DSL translation tasks.

**Objective 3: Cloud-Based API Infrastructure and Deployment**

Design and deploy a scalable, cloud-based API infrastructure supporting real-time interaction. Specific targets include: (a) implementing RESTful endpoints with comprehensive documentation following OpenAPI specifications; (b) integrating WebSocket protocols for real-time diagram updates during optimization; (c) deploying model inference endpoints on AWS SageMaker with appropriate scaling capabilities; and (d) achieving average response latencies below 2 seconds for standard diagram generation requests.

**Objective 4: System Integration and Performance Validation**

Integrate all system components into a cohesive platform and validate performance against established benchmarks. This includes: (a) implementing asynchronous task processing using Celery distributed task queue for computationally intensive operations; (b) conducting performance testing to verify system behavior under concurrent request scenarios; (c) establishing monitoring and logging infrastructure for production operations; and (d) documenting system architecture and deployment procedures for reproducibility.

---

## **4. PROBLEM STATEMENT**

Vietnamese secondary education faces significant challenges in geometry instruction that adversely affect student learning outcomes and engagement. This project addresses a multifaceted problem comprising educational, technical, and linguistic dimensions.

### **4.1 Educational Challenges**

Students frequently encounter difficulties in spatial visualization and geometric reasoning, skills essential for solving plane geometry problems. Research in mathematics education has identified that many students struggle to translate verbal problem descriptions into accurate geometric diagrams, leading to fundamental misunderstandings of problem requirements. Manual diagram construction, whether on paper or using general-purpose drawing software, is time-consuming and error-prone, often resulting in figures that fail to satisfy stated geometric constraints such as perpendicularity, parallelism, or length equality.

Furthermore, access to high-quality learning resources remains uneven. While printed textbooks provide basic problem sets, students lack access to interactive tools that can generate accurate diagrams automatically from problem specifications. The absence of immediate visual feedback mechanisms means students cannot efficiently validate their spatial understanding or identify specific conceptual gaps during independent study.

### **4.2 Technical Limitations of Existing Tools**

Current geometry software applications, while sophisticated, present several limitations that reduce their effectiveness in educational contexts. Tools such as GeoGebra require users to construct diagrams through sequences of discrete operations, demanding substantial prior knowledge of construction techniques. These systems do not automatically interpret formal problem specifications into diagrams, necessitating manual translation of textual descriptions into construction steps.

Additionally, existing tools lack integration with machine learning pipelines and do not provide infrastructure for training AI models on geometry understanding tasks. There is no existing solution that combines automatic diagram generation with dataset processing capabilities suitable for developing AI-powered geometry tutoring systems.

### **4.3 Computational Challenges**

From a computational perspective, automated geometry diagram generation presents several algorithmic challenges. The primary technical problem is constraint satisfaction: given a set of geometric specifications expressed in a formal language (e.g., "triangle ABC is isosceles with AB = AC", "point M is the midpoint of BC", "line AM is perpendicular to BC"), the system must compute point coordinates that simultaneously satisfy all constraints to a specified tolerance level.

This problem is nontrivial due to: (a) **Constraint interdependency** - modifications to satisfy one constraint may violate others, requiring global optimization; (b) **Non-convexity** - the constraint satisfaction problem typically features multiple local minima, making gradient-based optimization challenging; (c) **Degenerate configurations** - certain constraint combinations may have no valid solution or infinitely many solutions, requiring careful handling; (d) **Aesthetic considerations** - valid solutions should produce visually clear diagrams suitable for educational use, not merely mathematically correct configurations.

### **4.4 Natural Language Processing and Formalization Challenges**

The system must process geometry problems stated in Vietnamese natural language and convert them into formal geometric representations expressed in the DSL. This formalization task involves several challenges: (a) mapping Vietnamese geometric terminology to formal DSL predicates; (b) extracting structured constraint information from unstructured problem statements; (c) handling ambiguity and implied information in natural language descriptions; and (d) ensuring the generated DSL specifications are syntactically valid and semantically meaningful.

Training machine learning models to perform this Vietnamese-to-DSL translation requires substantial annotated data, which is currently unavailable in existing datasets. The project must therefore establish data processing pipelines to create such resources.

### **4.5 Problem Scope**

This project specifically addresses the problem of developing an end-to-end system that accepts formal geometric specifications as input and produces accurate diagram visualizations as output, while also providing the infrastructure necessary to train AI models that can translate Vietnamese geometry problems into formal representations. The scope encompasses plane geometry problems typically encountered in Vietnamese grades 9-12 curriculum, focusing on triangles, circles, quadrilaterals, and their properties including congruence, similarity, perpendicularity, parallelism, and metric relationships.

---

## **5. SIGNIFICANCE OF THE PROJECT**

This project contributes to both theoretical understanding and practical applications in the domains of educational technology, computational geometry, and natural language processing for low-resource languages.

### **5.1 Theoretical Contributions**

**Domain-Specific Language Design for Geometric Constraints**

This research introduces a novel DSL tailored specifically for representing geometric configurations in a form amenable to gradient-based optimization. Unlike existing geometry specification languages (such as those used in GeoGebra or Asymptote) which focus on imperative construction sequences, our DSL adopts a declarative constraint-based paradigm. This design facilitates automatic constraint solving through numerical optimization while maintaining human readability and mathematical expressiveness. The DSL syntax and semantics represent a contribution to research on domain-specific languages for mathematical problem solving.

**Hybrid Symbolic-Numeric Optimization Framework**

The system architecture demonstrates an integration of symbolic processing (DSL parsing and constraint extraction) with numeric optimization (PyTorch-based gradient descent). This hybrid approach bridges the gap between formal geometric reasoning and numerical computation, representing a methodological contribution to automated geometry processing and constraint satisfaction research. The framework shows how modern deep learning infrastructure can be repurposed for classical computational geometry problems.

**Vietnamese Geometry Dataset Development**

This project produces a dataset of Vietnamese plane geometry problems with formal DSL representations. The dataset includes natural language problem statements, formal DSL specifications, and diagram images. This resource addresses a gap in natural language processing research for Vietnamese mathematical text and enables future research in multilingual mathematical reasoning and low-resource language processing. The data processing pipeline itself represents a reusable contribution for similar dataset construction efforts.

### **5.2 Practical Benefits**

**Educational Impact**

For students, the platform provides immediate visual feedback on geometric configurations, supporting the development of spatial reasoning skills. By automating the diagram construction process from formal specifications, the system allows students to focus cognitive resources on conceptual understanding and problem-solving strategies rather than manual drawing mechanics. The platform can serve as a component in larger educational applications that provide interactive geometry learning experiences.

Teachers and educational content developers benefit from automated tools for creating problem sets with accompanying diagrams. The system's diagram generation capabilities enable efficient development of educational materials with geometrically accurate visualizations. Furthermore, the platform provides infrastructure that can be extended to build adaptive learning systems.

**Industry and Research Applications**

From a software engineering perspective, this project demonstrates best practices for deploying machine learning models in production environments using modern cloud infrastructure. The architecture illustrates design patterns for microservices-based AI applications, including API design, asynchronous task processing, model serving, and monitoring strategies. These implementation patterns are generalizable to other educational technology applications and AI-powered services.

The project showcases practical applications of large language models in structured prediction tasks, specifically for translating natural language into formal domain-specific languages. This extends understanding of LLM capabilities in mathematical reasoning domains and provides a reference implementation for similar applications.

**Open Source Contribution**

Upon completion, project components including the DSL specification, optimization algorithms, and API implementation will be made available. This enables the broader educational technology community to build upon this work, adapt the system for other languages or mathematical domains, and contribute improvements. The modular architecture facilitates reuse of individual components in other applications.

### **5.3 Societal Impact**

By improving accessibility to geometry education resources, this project contributes to educational equity. Students in regions with limited access to qualified mathematics teachers or expensive educational software can benefit from automated learning assistance tools built on this platform. The Vietnamese-language focus addresses the needs of approximately 95 million speakers, a population underserved by existing AI-powered educational tools that predominantly target English and other major languages.

Furthermore, successful demonstration of AI-assisted mathematics education may inspire similar efforts for other subjects and other low-resource languages, contributing to broader adoption of educational technology in developing regions.

---

## **6. PROJECT SCOPE AND LIMITATIONS**

### **6.1 Scope of Work**

This section delineates the boundaries of the project, specifying which features and functionalities are included within the current implementation.

**6.1.1 Geometric Content Coverage**

The system addresses plane geometry problems commonly encountered in Vietnamese secondary education (grades 9-12). Specifically, the implementation covers:

- **Triangles:** Equilateral, isosceles, right, scalene triangles and their properties including congruence criteria (SAS, ASA, SSS, AAS), similarity relationships, special points (centroid, orthocenter, incenter, circumcenter), and metric relationships
- **Circles:** Circle definitions, tangent lines and tangent circles, secants and chords, inscribed and circumscribed circles, angular relationships involving circles, and fundamental circle theorems
- **Quadrilaterals:** Parallelograms, rectangles, rhombi, squares, trapezoids, and their characteristic properties
- **Geometric Constructions:** Perpendicular and parallel line constructions, angle bisectors, perpendicular bisectors, midpoints, and projections
- **Geometric Relationships:** Distance equality, angle equality, parallelism, perpendicularity, collinearity, and concyclicity

**6.1.2 System Capabilities**

The implemented system provides the following functional capabilities:

- DSL-based diagram specification and parsing
- Constraint extraction and formalization from DSL expressions
- Numeric optimization for constraint satisfaction using gradient descent methods
- Diagram rendering with Matplotlib backend producing PNG and SVG outputs
- RESTful API endpoints for diagram generation and dataset operations
- WebSocket support for real-time diagram updates during optimization process
- Asynchronous task processing for computationally intensive operations using Celery
- Dataset processing pipeline including data loading, prompt generation, and model output formatting
- Vietnamese-to-DSL translation using fine-tuned language models
- Model training workflow with AWS SageMaker integration
- API documentation using OpenAPI/Swagger specifications
- Basic monitoring and logging infrastructure

**6.1.3 Deployment Infrastructure**

The system deployment encompasses:

- Containerized services using Docker
- AWS SageMaker endpoints for model inference
- Amazon S3 for dataset and model artifact storage
- Message queue implementation using Celery with Redis backend
- Development and staging environments
- Continuous integration pipeline for automated testing

### **6.2 Limitations and Exclusions**

This section explicitly identifies aspects not addressed in the current project implementation.

**6.2.1 Geometric Content Exclusions**

The following mathematical topics are explicitly excluded from the current scope:

- Three-dimensional geometry and solid figures
- Analytic geometry requiring coordinate systems as primary representation
- Conic sections (ellipses, parabolas, hyperbolas) and their properties
- Trigonometric identities and equations beyond basic angle relationships
- Geometric transformations (rotations, reflections, translations) as primary focus
- Advanced topics including inversive geometry, projective geometry, or non-Euclidean geometry

**6.2.2 System Feature Exclusions**

The implementation does not include:

- Automatic problem generation or synthesis capabilities (the system processes existing problems, not generates new ones)
- Automatic solution generation with step-by-step explanations
- User authentication and authorization systems (demonstration only includes basic session management)
- Persistent user account management and historical data storage
- Payment processing or subscription management capabilities
- Native mobile applications for iOS or Android platforms
- Real-time collaborative features allowing multiple users to work on the same problem simultaneously
- Gamification elements such as achievement systems, leaderboards, or progress tracking
- Adaptive learning algorithms that personalize problem difficulty based on user performance
- Voice input or accessibility features for visually impaired users
- Interactive diagram manipulation capabilities (users cannot drag points to modify diagrams)

**6.2.3 Production Deployment Limitations**

The current deployment does not encompass:

- Multi-region redundancy and geographic load balancing
- Comprehensive security auditing and penetration testing
- High-availability configurations with automatic failover
- Advanced caching strategies (CDN integration, distributed caching)
- Capacity planning for more than 1,000 concurrent users
- Compliance certifications (ISO, SOC2, GDPR)
- Professional SLA guarantees and 24/7 support infrastructure

### **6.3 Known Limitations**

**6.3.1 Technical Constraints**

Several technical limitations affect system performance and reliability:

**Optimization Convergence:** Diagrams involving more than 15 points or containing 10+ simultaneous constraints may fail to converge within the allocated 1,000 optimization epochs. In such cases, the system may produce diagrams that approximately satisfy constraints but with noticeable geometric inaccuracies. Complex cyclic constraint dependencies particularly challenge the gradient-based optimization approach.

**Language Model Accuracy:** The Vietnamese-to-DSL translation achieves approximately 85-90% accuracy on validation datasets. Failure modes include incorrect constraint interpretation, missing geometric objects, and malformed DSL syntax. Problems with ambiguous or uncommon phrasings are particularly susceptible to translation errors.

**Rendering Quality:** Matplotlib-based rendering produces functionally accurate diagrams but may lack aesthetic refinement compared to professionally designed educational materials. Label positioning occasionally results in text overlapping geometric elements, and line thickness/styling options are limited.

**Language Support:** The system exclusively processes Vietnamese problem statements. Extending to other languages would require collecting training data and fine-tuning models for each target language.

**6.3.2 Algorithmic Limitations**

The constraint optimization algorithm exhibits several known failure modes:

- **Degenerate Configurations:** Problems admitting degenerate solutions (e.g., collinear points that should form a triangle) may cause numerical instability
- **Multiple Solutions:** When problems have multiple valid diagram configurations, the system produces only one solution determined by initialization strategy
- **Local Minima:** Non-convex optimization landscape may trap solver in local minima, producing geometrically invalid configurations
- **Constraint Prioritization:** The system treats all constraints with equal weight; soft constraints or hierarchical constraint satisfaction are not supported

**6.3.3 Dataset Limitations**

The processed dataset comprises approximately 1,000-1,500 geometry problems with formal representations. This represents a relatively small corpus compared to large-scale machine learning datasets. This limited scale constrains model generalization capabilities, particularly for problem types underrepresented in training data. The dataset predominantly draws from specific Vietnamese educational publishers, potentially introducing stylistic biases.

**6.3.4 Resource Constraints**

Development operated under resource constraints typical of academic projects:

- **Computational Resources:** Training and inference utilize single-GPU instances; distributed training or multi-GPU inference parallelism not implemented
- **Budget:** AWS infrastructure operates within educational credits and free tier limitations, restricting instance types and storage capacity
- **Development Time:** Six-month project timeline limited scope of feature development and experimental iterations

### **6.4 Future Scope and Extensions**

While not implemented in the current project, several natural extensions merit consideration for future work:

- Automatic generation of new geometry problems (problem synthesis)
- Step-by-step solution generation with natural language explanations
- Extension to three-dimensional geometry problems
- Integration with handwriting recognition for problem statement input
- Interactive diagram manipulation with constraint preservation
- Support for additional languages through multilingual model training
- Enhanced visualization with dynamic geometry capabilities similar to GeoGebra
- Integration of formal proof verification systems
- Mobile application development for iOS and Android platforms
- Adaptive learning system that adjusts problem recommendations based on student performance

These extensions would enhance system capabilities but are considered beyond the scope of the current capstone project given time and resource constraints.

---

**End of Part I**
