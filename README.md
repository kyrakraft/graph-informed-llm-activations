# graph-informed-llm-activations
Exploring how different methods for providing graph-structured information to an LLM influence its activation space geometry, with the goal of facilitating steering.

## Overview

### Question
How do different methods for providing graph-structured information to an LLM influence its activation space? 

### Context
In certain cases, it is worth trading model generality for increased interpretability and control. In that vein, I am interested in exploring ways to nudge an LLM’s activations during reasoning tasks into a less complex and more orderly (and therefore, hopefully, more interpretable and steerable) geometric space. While this has been explored [1, 2, 3, 4], an analysis of approaches on graph-structured knowledge is limited. Questions include: How does the activation space change when knowledge is learned in-context vs. parametrically? What about when a model is trained on graph embeddings vs. a plain-text linearized graph? I will also track how qualities of the activation spaces correlate with LLM reasoning performance.

### Hypothesis
Methods incorporating knowledge into LLM parameters will yield more orderly activation spaces than in-context learning, because the activations will not be constrained by pre-learned weights blind to the KG. Within parametric learning, learning on graph embeddings (which have the graph’s structure explicitly built into them) will yield more orderly activation spaces than learning on the natural language embeddings comprising the linearized graph.  

### Architectures
I will compare the following knowledge injection approaches:
Baseline 1: LLM (~4-15B params?) with no external knowledge context
Baseline 2: LLM fine-tuned on an unstructured text description of the same topic as the KG
In-context learning via traditional GraphRAG: Retrieves relevant subgraph from the knowledge graph, and passes it to the LLM prompt as plain-text triples
Graph embedding soft-prompting: Pre-computes knowledge graph embeddings. During training, retrieves relevant graph embeddings and passes them to an MLP to translate them to the LLM’s representation space, and then projects the resulting soft-vector into the LLM’s input (such that the MLP learns, while the LLM weights remain frozen)
Prefix tuning: Appends vectors to the keys and values of the attention blocks at every hidden layer, and trains those vectors while keeping the original LLM weights frozen
Fine-tuning methods:
Linearized fine-tuning: Fine-tunes model on linearized plain-text version of graph
Representation alignment fine-tuning: Pre-computes knowledge graph embeddings. During training, as LLM learns on plain-text graph information, its hidden states are constrained via a contrastive loss objective to align with the graph embeddings (an MLP layer maps the graph embeddings into the LLM’s representation space so that they can be compared to the LLM’s internal states). The LLM weights thereby learn to map text embeddings into the geometric space of the graph
Additional approaches might involve combining in-context learning with #5 and #6. 

### Evaluation
For each knowledge injection approach, I will extract the LLM’s residual stream activations at a range of depths, over queries that require traversing the knowledge graph. For each resulting activation space, I will evaluate its geometric complexity and orderliness, potentially with metrics like intrinsic dimensionality, the space’s curvature (likely via a proxy), and others. Beyond evaluating the activation spaces in isolation, I will also evaluate the relative representational similarities of the respective activation spaces to the original graph. Some methods I am considering are 1. computing the above metrics on the original graph embeddings, and qualitatively comparing the graph’s representational geometry to the already-computed activation space’s representational geometry (not looking to find actual value matches, but rather general trends that indicate similarity relative to the baselines); 2. computing the alignment of relational displacement vectors by measuring the cosine similarity between an entity pair’s difference vector in the graph embedding space vs. their corresponding difference vector in the LLM’s activations; and 3. visually comparing the activations and original graph embeddings (e.g. with UMAP).


### References
[1] Gurnee, W., & Tegmark, M. (2023). Language models represent space and time. arXiv preprint arXiv:2310.02207.
[2] Sarfati, R., Liu, T. J. B., Boullé, N., & Earls, C. J. (2025). Lines of thought in large language models. International Conference on Learning Representations (ICLR).
[3] Bigelow, E., Sarfati, R., Wurgaft, D., Lewis, O., McGrath, T., Merullo, J., Geiger, A., & Lubana, E. S. (2026). Stories in space: In-context learning trajectories in conceptual belief space. arXiv preprint arXiv:2605.12412.
[4] Zhao, J., Yang, Y., Hu, X., Tong, J., Lu, Y., Wu, W., Gui, T., Zhang, Q., & Huang, X. (2025). Understanding parametric and contextual knowledge reconciliation within large language models. Advances in Neural Information Processing Systems (NeurIPS), 38. 
