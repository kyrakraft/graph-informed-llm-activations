# graph-informed-llm-activations
Exploring how different methods for providing graph-structured information to an LLM influence its activation space geometry (with activation steering in mind for later).

## Overview of Research Plan

### Primary Question
How do different methods for providing graph-structured information to an LLM influence its activation space? 

### Context
I am interested in exploring ways to nudge an LLM’s activations during reasoning tasks into a less complex and more orderly geometric space. While this has been explored [1, 2, 3, 4], an analysis of approaches on graph-structured knowledge is limited.

Motivation: My high-level hope is that graph-structured knowledge, which is highly structured, will make an LLM's knowledge representation interpretable and steerable.

Questions include:
- How does the activation space change when knowledge is learned in-context vs. parametrically?
- What about when a model is trained on graph embeddings vs. a plain-text linearized graph? 

I will also track how qualities of the activation spaces correlate with LLM reasoning performance.

### Hypotheses
Methods incorporating knowledge into LLM parameters will yield more orderly activation spaces than in-context learning, because the activations will not be constrained by pre-learned weights blind to the KG.

Within parametric learning, learning on graph embeddings (which have the graph’s structure explicitly built into them) will yield more orderly activation spaces than learning on the natural language embeddings comprising the linearized graph because the graph's structure is explicitly accounted for in graph embeddings themselves.

### Architectures
I will compare the following knowledge injection approaches:

- Baseline 1: LLM-only (likely one with ~4-15B params), with no external knowledge context
- Baseline 2: LLM fine-tuned on an unstructured text description of the same topic as the KG
- In-context learning via traditional GraphRAG: Retrieves relevant subgraph from the knowledge graph, and passes it to the LLM prompt as plain-text triples
- Graph embedding soft-prompting: Pre-computes knowledge graph embeddings. During training, retrieves relevant graph embeddings and passes them to an MLP to translate them to the LLM’s representation space, and then projects the resulting soft-vector into the LLM’s input (such that the MLP learns, while the LLM weights remain frozen)
- Prefix tuning: Appends vectors to the keys and values of the attention blocks at every hidden layer, and trains those vectors while keeping the original LLM weights frozen
- Fine-tuning methods:
    - Linearized fine-tuning: Fine-tunes model on linearized plain-text version of graph
    - Representation alignment fine-tuning: Pre-computes knowledge graph embeddings. During training, as LLM learns on plain-text graph information, its hidden states are constrained via a contrastive loss objective to align with the graph embeddings (an MLP layer maps the graph embeddings into the LLM’s representation space so that they can be compared to the LLM’s internal states). The LLM weights thereby learn to map text embeddings into the geometric space of the graph
Additional approaches might involve combining in-context learning with #5 and #6. 

### Evaluation
For each knowledge injection approach, I will extract the LLM’s residual stream activations at a range of depths, over queries that require traversing the knowledge graph. 

For each resulting activation space, I will evaluate its geometric complexity and orderliness, potentially with metrics like intrinsic dimensionality, the space’s curvature (likely via a proxy), and others. Beyond evaluating the activation spaces in isolation, I will also evaluate the relative representational similarities of the respective activation spaces to the original graph.

Some methods I am considering are 1. computing the above metrics on the original graph embeddings, and qualitatively comparing the graph’s representational geometry to the already-computed activation space’s representational geometry (not looking to find actual value matches, but rather general trends that indicate similarity relative to the baselines); 2. computing the alignment of relational displacement vectors by measuring the cosine similarity between an entity pair’s difference vector in the graph embedding space vs. their corresponding difference vector in the LLM’s activations; and 3. visually comparing the activations and original graph embeddings (e.g. with UMAP).

### References
[1] Gurnee, W., & Tegmark, M. (2023). Language models represent space and time. arXiv preprint arXiv:2310.02207.

[2] Sarfati, R., Liu, T. J. B., Boullé, N., & Earls, C. J. (2025). Lines of thought in large language models. International Conference on Learning Representations (ICLR).

[3] Bigelow, E., Sarfati, R., Wurgaft, D., Lewis, O., McGrath, T., Merullo, J., Geiger, A., & Lubana, E. S. (2026). Stories in space: In-context learning trajectories in conceptual belief space. arXiv preprint arXiv:2605.12412.

[4] Zhao, J., Yang, Y., Hu, X., Tong, J., Lu, Y., Wu, W., Gui, T., Zhang, Q., & Huang, X. (2025). Understanding parametric and contextual knowledge reconciliation within large language models. Advances in Neural Information Processing Systems (NeurIPS), 38. 


## Notes, open questions / considerations, additional thoughts

1. What knowledge graph makes the most sense to use? Here are some of my considerations:
    - Argument a) On one hand, maybe it makes sense to use a KG where the graph structure is not inferable from the semantic information alone. For example, a graph representing lexical taxonomy formalizes a structure that is likely already implicitly represented in the LLM. Would this render “Baseline 1,” an LLM with no supplemental knowledge, flawed (as the LLM would already implicitly contain some of the structural knowledge I am aiming to test the effects of providing it with) and also potentially decrease the gap between in-context and parametric learning methods?
    - Argument b) On the other hand, the question of how explicit formalization of structure can change the representation space (beyond whatever intrinsic domain structure might be learned implicitly during pre-training) is an interesting thing to test, too. In fact, maybe this is actually more informative.
    - My current thinking is that (b) is a strong enough argument to warrant using a KG that contains publicly known information. In terms of specific knowledge graphs, I’m considering using a subset of WikiData due to its customizability (I could use a subgraph with both general and niche knowledge).
2. Graph embedding method?
   - Different graph embedding methods would yield different activations.
4. Thoughts on potential evaluation metrics
   - Are these the right metrics?
5. Limitations
   - Various potential confounders...e.g., even if some of the activations do see this result, it does not necessarily mean that the cause is the structure of the graph. 
7. Other thoughts
    - Also would be interesting to see how this changes with different LLM sizes, knowledge graph size, type of knowledge graph, and training times and inference times. However, best to save these quesitons for a follow-up project to ensure a narrow-enough scope for this project.
    - Could also potentially train SAEs on activations...
