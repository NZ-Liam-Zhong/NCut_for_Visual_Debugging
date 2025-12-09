# Tutorials For Visual Debugging Using Nystrom Ncut

## Introduction

25 years ago, Jianbo Shi introduced **Normalized Cuts** (spectral clustering), a graph-theoretic approach to perceptual grouping that became a staple in unsupervised image segmentation. While the original method was powerful, it was computationally heavy for large-scale data.

Today, we are introducing **Nyström Normalized Cuts**, a modern evolution of our original work designed for the deep learning era. By leveraging the Nyström approximation, we have reduced the complexity to **$O(n)$ time** and **$O(1)$ space**. 

This massive speedup allows us to solve million-scale graphs in mere milliseconds, making it possible to run spectral analysis on high-resolution feature maps in real-time. It effectively turns the classic Ncut into a lightweight, interactive probe for visual debugging, helping researchers peek inside the "black box" of model backbones (like DINO, SAM, CLIP, LLAMA, GPT2, etc) to see exactly how the model groups and understands visual/text concepts. (Link: https://ncut-pytorch.readthedocs.io). In this tutorial, we will introduce how NCut for visual debugging works.


## What is Visual Debugging?

Foundation models, such as LLaMA, are prone to hallucinations. A classic example is showing a model an image of a hand with five fingers and asking, "How many fingers?"—the model might confidently answer "four" or "six."

<img width="476" height="312" alt="image" src="https://github.com/user-attachments/assets/d67012e9-6a66-4f3f-b68a-56b6e473ccb8" />
Human: How many fingers are there?
AI(LLAVA): There are four fingers in this image.

Traditionally, we might attribute this failure to insufficient training data or a need for more epochs. However, **Visual Debugging** offers a more granular diagnosis. By applying NCut to visualize the internal feature representations layer by layer, we can pinpoint exactly *where* the model "loses sight" of the concept. We might observe that in early layers, the five distinct fingers are clearly separated in the spectral embedding, but in deeper layers, the features blur or merge, leading to the incorrect output.



## How to do visual debugging

1. Curate a batch of similar photos that expose the suspected failure mode, then run batched Nyström NCut on that batch. For example, if LLaVA miscounts fingers, collect hands with different finger counts and segment them together; inspect how cluster colors change under these controlled variations. <br>

<img width="2688" height="1680" alt="image" src="https://github.com/user-attachments/assets/d1dd2903-daa8-4c39-b5e9-994754015f02" />

Human: How many fingers are there? <br>
AI:<br>
Image 1: The image shows a hand with five fingers.
Image 2: The image shows a hand with four fingers extended.
Image 3: The image shows a hand with five fingers.
Image 4: The image shows a hand with five fingers.
Image 5: There are five fingers in the image.
Image 6: The image shows a hand with five fingers.
Image 7: The image shows a hand with five fingers.
Image 8: The image shows a hand with five fingers.
Image 9: The image shows a person's hands with their fingers spread out. However, without more specific information, it is not possible to determine the exact number of fingers in the image.
Image 10: There are a total of 16 fingers in the image.
Image 11: The image shows a hand with five fingers.
Image 12: The image shows a hand with four fingers extended upwards.
Image 13: The image shows two hands, each with five fingers. So, there are a total of 10 fingers in the image.
Image 14: The image shows two hands reaching towards each other, but the exact number of fingers is not visible.
Image 15: The image shows a hand with five fingers.
Image 16: The image shows two hands with a total of ten fingers.
Image 17: There are a total of 10 fingers in the image, with each hand having five fingers.
Image 18: In the image, there are a total of 12 fingers visible.
Image 19: There are ten fingers in this image.
Image 20: There are a total of 10 fingers in the image.
Image 21: In the image, there are four fingers visible.
Image 22: In the image, there are a total of 10 fingers visible.
Image 23: The image shows a hand with four fingers extended.
Image 24: The image shows two hands, each with five fingers. So, there are a total of ten fingers in the image.
Image 25: The image shows a hand with five fingers.
Image 26: The image shows a hand with five fingers.
Image 27: The image shows a hand with five fingers.
Image 28: There are five fingers in this image.
Image 29: There are four fingers in this image.
Image 30: The image shows a hand with five fingers.
Image 31: The image shows a hand with five fingers.
Image 32: The image shows a hand with four fingers extended.
Image 33: The image shows a hand with five fingers.
Image 34: The image shows a hand with four fingers.
Image 35: The image shows two hands clasped together, which means there are a total of 10 fingers.
Image 36: The image shows a hand with five fingers.
Image 37: The image shows a hand with five fingers.
Image 38: The image shows a hand with five fingers.
Image 39: There are five fingers in this image.
Image 40: The image shows a hand with five fingers.


2. Analyze per layer with the entire batch side by side. If a small perturbation in one photo causes color shifts only in that case, the model is highly sensitive to that factor. If colors stay stable across the batch despite perturbations, the model is insensitive to that semantic cue. <br><br>

First, we can visualize the last two layers of encoder of LLAVA.<br>
Layer 31<br>
<img width="3545" height="2212" alt="image" src="https://github.com/user-attachments/assets/e9cf2a88-1f1f-4bd6-9848-58e89fe5cc4f" /><br>
Layer 30<br>
<img width="3545" height="2212" alt="image" src="https://github.com/user-attachments/assets/c73a3b86-b54f-4340-8d21-84f68590db50" /><br>
It seems that the colour of fingers or hands both won't change in different images. But we still don't know what happen and what makes the AI answer wrongly in different settings. It seems that we should go in to details of the vision tower.



3. Trace across layers (and submodules when the model has multiple parts). For multi-branch systems like LLaVA, separately inspect the vision tower and the vision-language backbone to localize which component fails to preserve the relevant structure.<br>

LLAVA has a 3-crop process in vision tower. The image below is the visualization results of layer 15 (Image 1~3 are from the 1th hand image, Image 4~6 are from the 2th hand image, ...) <br>
<img width="3549" height="6680" alt="image" src="https://github.com/user-attachments/assets/97879d74-9942-4cde-aa62-35425e3c8429" /><br>
Compared with the LLAVA's text answers above, we can see that the changes of colour contribute mostly to the change of text. And it's the gestures of hands instead of the number of fingers of the hand makes the changes in colour. 





