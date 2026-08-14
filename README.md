# graph-informed-llm-activations
Exploring how different methods for providing graph-structured information to an LLM influence its activation space geometry (with activation steering in mind for later).

My hope is that graph-structured knowledge, which is highly structured, will make an LLM's knowledge representation interpretable and steerable.

Currently in exploratory phase. General plan below, subject to change based on exploration results.

I plan to use Gemma-2 (eventually 9B, with 2B as a smaller comparison point. Currently using 2B.) with LoRA for all fine-tuning conditions.

## Overview of Research Plan

### Primary Question
How do different methods for providing graph-structured information to an LLM influence its activation space? 

### Context
I am interested in exploring ways to nudge an LLM’s activations during reasoning tasks into a less complex and more orderly geometric space. While this has been explored [1, 2, 3, 4], an analysis of approaches on graph-structured knowledge is limited. Additionally, the question of whether it matters how the graph-structured knowledge is given to the model (in-context vs. through its parameters, as text vs. as graph embeddings) is largely unexplored.

The closest existing work is Park et al. [5], who found that when a model is given enough in-context information specifying a novel graph structure over familiar concepts, its representations reorganize away from their pretrained semantics to align with the context-specified structure. [11, 12] contest this claim, arguing that Park et al.'s task can be solved by induction circuits, and that the graph-shaped geometry is a byproduct of previous-token heads mixing each token's representation with its predecessor (which, in a random walk, is always a graph neighbor).

Notably, both sides of this debate concern in-context learning only; neither tested fine-tuned parameters. This presents a gap – a fine-tuned model has no context to mix at query time, so if graph-aligned geometry appears in parametric conditions, the mixing explanation is unavailable. Either way, the parametric side of this debate is unmeasured, and this project measures it.


Questions include:
- How does the activation space change when knowledge is learned in-context vs. parametrically?
- What about when a model is trained on graph embeddings vs. a plain-text linearized graph? 
- Is there a data-scale threshold for parametric reorganization analogous to the context-scale threshold Park et al. found for in-context learning?

I will also track how qualities of the activation spaces correlate with LLM reasoning performance.

### Hypotheses
Methods incorporating knowledge into LLM parameters will yield more orderly activation spaces than in-context learning, because the activations will not be constrained by pre-learned weights blind to the KG.

Within parametric learning, learning on graph embeddings (which have the graph’s structure explicitly built into them) will yield more orderly activation spaces than learning on the natural language embeddings comprising the linearized graph because the graph's structure is explicitly accounted for in graph embeddings themselves.

## Experimental setup

### Knowledge graph design

I will have real concepts as nodes (cities, people), natural relation types ("trades-with," "is-allied-with"), but edges assigned arbitrarily so the topology cannot be predicted from world knowledge. This is because known facts would not give fine-tuning enough novel information to learn. Using real entities (vs. invented tokens) will let me test the effect of formalized structure on top of existing pretrained knowledge about those concepts.

### Knowledge representation formats and learning methods
The design has two axes: knowledge "learning" method (in-context vs. parametric) and knowledge representation format (linearized triples vs. graph embeddings).

I will compare the following setups:

- Baseline 1: LLM-only, with no external knowledge context
- Baseline 2: LLM fine-tuned on an unstructured text description of the same facts as the KG triplets, so content is the same and only the formatting differs.
- In-context learning via traditional GraphRAG: Retrieves relevant subgraph from the knowledge graph, and passes it to the LLM prompt as plain-text triples
- Graph embedding soft-prompting: Pre-computes knowledge graph embeddings. During training, retrieves relevant graph embeddings and passes them to an MLP to translate them to the LLM’s representation space, and then projects the resulting soft-vector into the LLM’s input (such that the MLP learns, while the LLM weights remain frozen)
- Fine-tuning methods:
    - Linearized fine-tuning: Fine-tunes model on linearized plain-text version of graph
    - Representation alignment fine-tuning: Pre-computes knowledge graph embeddings. During training, as LLM learns on plain-text graph information, its hidden states are constrained via a contrastive loss objective to align with the graph embeddings (an MLP layer maps the graph embeddings into the LLM’s representation space so that they can be compared to the LLM’s internal states). The LLM weights thereby learn to map text embeddings into the geometric space of the graph

### Evaluation
For each approach, I will extract the LLM’s residual stream activations at a range of depths, over queries that require traversing the knowledge graph. 

I will use identical query strings across all conditions and extract residual stream activations only at matched query-token positions (e.g., the last token of each entity in the query, and the final pre-answer token), and not at positions within the injected context. Activations will be mean-centered per condition before computing geometric metrics, and all metrics will be computed over the same fixed set of queries. That query set will include entity pairs spanning the full range of graph distances. This is in part because directly connected entities co-occur within a single line of the prompt (and co-occurring tokens can acquire similar activations for purely contextual reasons), and pairs two or more hops apart never co-occur in a line, so activation similarity that gradates with multi-hop graph distance indicates the model is likely using the structure of the graph. 

Relatedly, the test queries themselves will be held out and compositional, i.e. multi-hop questions whose entity pairs never co-occur in any single training or context line. This is so that answering correctly depends on actually composing the graph's structure rather than on surface co-occurrence.

I will then evaluate the following:

1. Geometric complexity and orderliness: intrinsic dimensionality, the space's curvature (likely via a proxy), and others. Prior work [6, 7] shows that intrinsic dimensionality tends to expand in early transformer layers, peak, then contract in later layers (with the most semantically rich information concentrating where ID contracts). So, I will compare these metrics layerwise with this expected profile in mind, rather than as single numbers. I’d hypothesize that learning methods that "work" (see high performance) will increase the late-layer ID contraction on graph-relevant queries.

2. Representational similarity to the original graph: I will quantify the alignment between each activation space and the graph embedding space using representational similarity measures. Since the two spaces have different dimensionalities and arbitrary orientations, they cannot be compared coordinate-wise; these methods instead compare structural similarity over the same set of entities. RSA correlates the two spaces' pairwise-distance matrices (entities close in the graph should be close in activations); CKA [10] does the same via Gram matrices, invariant to rotation and scaling; Procrustes analysis finds the best rotation mapping one space onto the other and measures residual error. I will also look at relational displacement vectors (with a caveat – Hernandez et al. [8] found that relation decoding in LLMs is often better described by per-relation affine maps than by a single shared offset vector, so raw difference vectors may be too coarse on the LLM side....). On the graph side, though, TransE models each relation exactly as a displacement, which is part of why I chose it. So, I will start by measuring the cosine similarity between an entity pair's difference vector in the graph embedding space and the corresponding difference vector in the activations, and upgrade to estimating per-relation linear maps in the style of Hernandez et al. if the raw vectors end up being too coarse. UMAP visualizations of the activations vs. graph embeddings might be included as illustrations.

3. Functional decodability via linear probes: The geometry metrics in #1 measure how simple the activation space is, and this axis measures whether the graph's structure is accessible from that activation space. For each condition and layer, I will train linear probes on the extracted activations to predict properties of the graph (e.g., relation type, hop distance from the query entity, neighbor identity). Most current interpretability and steering methods (steering vectors, probing, model editing, SAE features) assume features are roughly linear directions, and the model itself reads the residual stream through linear maps. That said, I will also train small MLP probes on the same tasks, and the gap between MLP and linear accuracy will measure how much information is present but not linearly accessible.


Alongside all of the above, I will track each condition's task accuracy on multi-hop queries over the KG, so that geometric properties can be correlated with reasoning performance.

Additionally, to eliminate some potential confounders, will do:

- Shuffled graph, aka in-context learning condition re-run with edges permuted (same length, format, entities, relations). Separates format effects from correct-edge effects.

- Neighbor-mixing, aka random vectors averaged along the true graph's edges (normalized adjacency), to address the concern raised in [11] that graph-shaped activation geometry can arise purely from previous-token heads blending each entity with its in-context neighbors, rather than from any learned representation of the graph.


### Extensions (if time permits):

- Graph size / training data scaling. Park et al. found a sudden reorganization of representations as context scales...the analogous question for parametric learning (“is there a data-or-steps threshold at which fine-tuning reorganizes representations?”) could be probed by varying graph size or training duration.
- Model size (e.g. comparing results between Gemma-2 2B and 9B)
- SAE analysis to examine which sparse features activate on graph entities across conditions
- Combining in-context learning with the fine-tuning methods
- Head-ablation test, extending [11, 12]. Identify previous-token heads and zero-ablate the top-k in both the in-context and linearized fine-tuning conditions. If ablation substantially eliminates graph alignment in the in-context condition but not in the fine-tuned condition, this would directly support the idea that in-context geometry is just an artifact of previous-token mixing while parametric geometry is not.




### References
[1] Gurnee, W., & Tegmark, M. (2023). Language models represent space and time. arXiv preprint arXiv:2310.02207.

[2] Sarfati, R., Liu, T. J. B., Boullé, N., & Earls, C. J. (2025). Lines of thought in large language models. International Conference on Learning Representations (ICLR).

[3] Bigelow, E., Sarfati, R., Wurgaft, D., Lewis, O., McGrath, T., Merullo, J., Geiger, A., & Lubana, E. S. (2026). Stories in space: In-context learning trajectories in conceptual belief space. arXiv preprint arXiv:2605.12412.

[4] Zhao, J., Yang, Y., Hu, X., Tong, J., Lu, Y., Wu, W., Gui, T., Zhang, Q., & Huang, X. (2025). Understanding parametric and contextual knowledge reconciliation within large language models. Advances in Neural Information Processing Systems (NeurIPS), 38. 

[5] Park et al. (2024). In-Context Learning of Representations.

[6] Ansuini et al. (2019). Intrinsic Dimension of Data Representations in Deep Neural Networks. NeurIPS.

[7] Valeriani et al. (2023). The Geometry of Hidden Representations of Large Transformer Models. NeurIPS.

[8] Hernandez et al. (2024). Linearity of Relation Decoding in Transformer Language Models. ICLR.

[9] Lieberum et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.

[10] Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. ICML.

[11] In-context learning of representations can be explained by induction circuits. ICLR 2026 Blogposts Track.

[12] Lepori et al. (2026).