# Notes

## Initial setup
gemma-2-2b base predicts 'a' for prompt 'capital of France is'. TransformerLens matches HuggingFace for this so TL pipeline verified.

Diffuse distribution. top 5:
' a' 0.202
' the' 0.095
' one' 0.074
' also' 0.074
' home' 0.058

## Testing w small kg

KG relations chosen to be plausible for arbitrary entity pairs, and at most weakly determined by pretraining, such that the model has no confident prior for or against any particular edge. This is so that KG-information neither "leaks from" nor conflicts with common knowledge.

Tried one pass of 60 triples, 4 mentions per entity. no visible reorganization in PCA, which... is not surprising.
Did PCA variance ratio though: PC1+2 capture <30% at early/mid layers so the lots likely don't tell us much anyway; layer 25 has one direction with 81% of variance (possibly due to anisotropy?). 
Also did RSA for each layer...rho was close to 0 for every layer.

Next steps:
May switch to newer LM like qwen...

Going to switch to a different KG at the "edge of knowledge" (so that the structure is not already known from pretraining) like this one: https://psychedelicskg.com/. Will also do some held-out edges where success depends on graph structure rather than term co-occurrence, with ablation/shuffling experiment.

