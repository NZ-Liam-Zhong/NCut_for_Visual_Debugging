# Tutorials For Visual Debugging Using Nystrom Ncut

## Introduction

25 years ago, Jianbo Shi introduced **Normalized Cuts** (spectral clustering), a graph-theoretic approach to perceptual grouping that became a staple in unsupervised image segmentation. While the original method was powerful, it was computationally heavy for large-scale data.

Today, we are introducing **Nyström Normalized Cuts**, a modern evolution of our original work designed for the deep learning era. By leveraging the Nyström approximation, we have reduced the complexity to **$O(n)$ time** and **$O(1)$ space**. 

This massive speedup allows us to solve million-scale graphs in mere milliseconds, making it possible to run spectral analysis on high-resolution feature maps in real-time. It effectively turns the classic Ncut into a lightweight, interactive probe for visual debugging, helping researchers peek inside the "black box" of model backbones (like DINO, SAM, CLIP, LLAMA, GPT2, etc) to see exactly how the model groups and understands visual/text concepts. (Link: https://ncut-pytorch.readthedocs.io). In this tutorial, we will introduce how NCut for visual debugging works.

## Why Visual Debugging with NCut?

Our approach relies on three key observations regarding modern deep features:

1. **The NCut Scaling Law**: We observe a distinct scaling behavior in spectral analysis—as we visualize larger sets of features jointly, the underlying semantic structures manifest more clearly. What's more, we utilize **Aligned Cut**, which allows us to align the feature spaces across different layers or even different models. This creates a unified visual coordinate system, making comparisons intuitive and mathematically consistent.

2. **Resilience of Transformer Representations**: Most current foundation models rely on Transformer architectures, which exhibit a remarkable property of self-correction. As demonstrated in works like *InstructPix2Pix*, it is possible to swap or perturb the intermediate feature space without causing a total collapse; the subsequent network layers often recover the semantic flow. This resilience allows us to intervene and visualize internal states dynamically without breaking the model's fundamental operation.

3. **Universal Structure, Distinct Focus**: While there is a strong "universal" feature space shared across models, significant nuances remain. Different models—or even different layers within the same model—possess their own "feature personalities." NCut allows us to disentangle these, showing that while models may agree on the broad strokes, they often diverge in what they prioritize—be it texture, geometry, or high-level semantic abstractions.

## What is Visual Debugging?

Foundation models, such as LLaMA, are prone to hallucinations. A classic example is showing a model an image of a hand with five fingers and asking, "How many fingers?"—the model might confidently answer "four" or "six."

Traditionally, we might attribute this failure to insufficient training data or a need for more epochs. However, **Visual Debugging** offers a more granular diagnosis. By applying NCut to visualize the internal feature representations layer by layer, we can pinpoint exactly *where* the model "loses sight" of the concept. We might observe that in early layers, the five distinct fingers are clearly separated in the spectral embedding, but in deeper layers, the features blur or merge, leading to the incorrect output.


## How to do Visual Debugging?


