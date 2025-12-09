# Tutorials For Visual Debugging Using Nystrom Ncut

## Introduction

25 years ago, Jianbo Shi introduced **Normalized Cuts** (spectral clustering), a graph-theoretic approach to perceptual grouping that became a staple in unsupervised image segmentation. While the original method was powerful, it was computationally heavy for large-scale data.

Today, we are introducing **Nyström Normalized Cuts**, a modern evolution of our original work designed for the deep learning era. By leveraging the Nyström approximation, we have reduced the complexity to **$O(n)$ time** and **$O(1)$ space**. 

This massive speedup allows us to solve million-scale graphs in mere milliseconds, making it possible to run spectral analysis on high-resolution feature maps in real-time. It effectively turns the classic Ncut into a lightweight, interactive probe for visual debugging, helping researchers peek inside the "black box" of model backbones (like DINO, SAM, CLIP, LLAMA, GPT2, etc) to see exactly how the model groups and understands visual/text concepts. (Link: https://ncut-pytorch.readthedocs.io). In this tutorial, we will introduce how NCut for visual debugging works.


## What is Visual Debugging?

Foundation models, such as LLaMA, are prone to hallucinations. A classic example is showing a model an image of a hand with five fingers and asking, "How many fingers?"—the model might confidently answer "four" or "six."

<img width="476" height="312" alt="image" src="https://github.com/user-attachments/assets/d67012e9-6a66-4f3f-b68a-56b6e473ccb8" /><br>
Human: How many fingers are there?<br>
AI(LLAVA): There are four fingers in this image.<br>

Traditionally, we might attribute this failure to insufficient training data or a need for more epochs. However, **Visual Debugging** offers a more granular diagnosis. By applying NCut to visualize the internal feature representations layer by layer, we can pinpoint exactly *where* the model "loses sight" of the concept. We might observe that in early layers, the five distinct fingers are clearly separated in the spectral embedding, but in deeper layers, the features blur or merge, leading to the incorrect output.



## How to do visual debugging

1. Curate a batch of similar photos that expose the suspected failure mode, then run batched Nyström NCut on that batch. For example, if LLaVA miscounts fingers, collect hands with different finger counts and segment them together; inspect how cluster colors change under these controlled variations. <br>

<img width="2688" height="1680" alt="image" src="https://github.com/user-attachments/assets/d1dd2903-daa8-4c39-b5e9-994754015f02" /><br>

Human: How many fingers are there? <br>
AI: <br>
Image 1: The image shows a hand with five fingers.<br>
Image 2: The image shows a hand with four fingers extended.<br>
Image 3: The image shows a hand with five fingers.<br>
Image 4: The image shows a hand with five fingers.<br>
Image 5: There are five fingers in the image.<br>
Image 6: The image shows a hand with five fingers.<br>
Image 7: The image shows a hand with five fingers.<br>
Image 8: The image shows a hand with five fingers.<br>
Image 9: The image shows a person's hands with their fingers spread out. However, without more specific information, it is not possible to determine the exact number of fingers in the image.<br>
Image 10: There are a total of 16 fingers in the image.<br>
Image 11: The image shows a hand with five fingers.<br>
Image 12: The image shows a hand with four fingers extended upwards.<br>
Image 13: The image shows two hands, each with five fingers. So, there are a total of 10 fingers in the image.<br>
Image 14: The image shows two hands reaching towards each other, but the exact number of fingers is not visible.<br>
Image 15: The image shows a hand with five fingers.<br>
Image 16: The image shows two hands with a total of ten fingers.<br>
Image 17: There are a total of 10 fingers in the image, with each hand having five fingers.<br>
Image 18: In the image, there are a total of 12 fingers visible.<br>
Image 19: There are ten fingers in this image.<br>
Image 20: There are a total of 10 fingers in the image.<br>
Image 21: In the image, there are four fingers visible.<br>
Image 22: In the image, there are a total of 10 fingers visible.<br>
Image 23: The image shows a hand with four fingers extended.<br>
Image 24: The image shows two hands, each with five fingers. So, there are a total of ten fingers in the image.<br>
Image 25: The image shows a hand with five fingers.<br>
Image 26: The image shows a hand with five fingers.<br>
Image 27: The image shows a hand with five fingers.<br>
Image 28: There are five fingers in this image.<br>
Image 29: There are four fingers in this image.<br>
Image 30: The image shows a hand with five fingers.<br>
Image 31: The image shows a hand with five fingers.<br>
Image 32: The image shows a hand with four fingers extended.<br>
Image 33: The image shows a hand with five fingers.<br>
Image 34: The image shows a hand with four fingers.<br>
Image 35: The image shows two hands clasped together, which means there are a total of 10 fingers.<br>
Image 36: The image shows a hand with five fingers.<br>
Image 37: The image shows a hand with five fingers.<br>
Image 38: The image shows a hand with five fingers.<br>
Image 39: There are five fingers in this image.<br>
Image 40: The image shows a hand with five fingers.<br>


2. Analyze per layer with the entire batch side by side. If a small perturbation in one photo causes color shifts only in that case, the model is highly sensitive to that factor. If colors stay stable across the batch despite perturbations, the model is insensitive to that semantic cue. First, we can visualize the  results of layer 4 of vision-language backbone of LLAVA (the text tokens can get information from the image tokens in every single layer of vision-language backbone, not necessarily only the last feature).<br>
<img width="3545" height="2212" alt="image" src="https://github.com/user-attachments/assets/d64c6469-ac80-49eb-9348-f0e492d02cfe" /><br>
Compared with the LLAVA's text answers below, we can see that the changes of colour contribute mostly to the change of text. And it's the gestures of hands instead of the number of fingers of the hand makes the changes in colour. 
### Finger numbers predicted by AI
| Row \ Col | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01-08** | 5 Fingers | 4 Extended | 5 Fingers | 5 Fingers | 5 Fingers | 5 Fingers | 5 Fingers | 5 Fingers |
| **09-16** | Unknown | 16 Fingers | 5 Fingers | 4 Upwards | 10 (2 Hands) | Unknown | 5 Fingers | 10 Total |
| **17-24** | 10 Total | 12 Visible | 10 Fingers | 10 Fingers | 4 Visible | 10 Visible | 4 Extended | 10 (2 Hands)|
| **25-32** | 5 Fingers | 5 Fingers | 5 Fingers | 5 Fingers | 4 Fingers | 5 Fingers | 5 Fingers | 4 Extended |
| **33-40** | 5 Fingers | 4 Fingers | 10 (Clasped)| 5 Fingers | 5 Fingers | 5 Fingers | 5 Fingers | 5 Fingers |


3. Trace across layers (and submodules when the model has multiple parts). For multi-branch systems like LLaVA, separately inspect the vision tower and the vision-language backbone to localize which component fails to preserve the relevant structure.<br>

LLAVA has a 3-crop process in vision tower (Image 1,2,3 are all from the 1th hand picture, Image 4,5,6 are all from the 2nd hand picture, ...). The image below is the visualization results of the last layer of vision tower. We can see that even the output of vision tower doesn't have significant colour changes relevant with the numbers of the fingers. <br>
<img width="3549" height="6680" alt="image" src="https://github.com/user-attachments/assets/47e1c945-afed-4c55-82f6-29452f97d6bb" />
<br>








