# <h1 align="center">**Computer Vision**</h1>

<p align="center">
<img src="ComputerVision.png"> 
</p>

This repository contains a collection of links to my repositories showcasing implementations of Computer Vision models in Python. It features several basic models using [Convolutional Neural Networks (CNNs)](https://developers.google.com/machine-learning/glossary/#convolutional_neural_network). CNNs are a type of neural network designed to process grid-structured data like images and are particularly effective at recognizing visual patterns by capturing local features through convolutions.

Additionally, it includes models applying [Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning), a technique that reuses pre-trained models on large datasets, allowing for high performance with lower computational costs. This technique leverages the knowledge acquired by models trained on millions of images or videos for specific new tasks, enhancing efficiency and accuracy.

Furthermore, several notebooks are implemented where [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) models are fine-tuned. ViTs use attention mechanisms to process images as a whole, outperforming state-of-the-art convolutional networks and requiring fewer computational resources for training. These models excel at capturing long-range dependencies in input data, resulting in a better understanding of the global structures in images.

## **What is Computer Vision?**

Computer Vision is a field of artificial intelligence that enables machines to interpret and understand the visual world through images and videos. It employs algorithms and deep learning models to perform tasks such as object recognition, action detection, and image segmentation. Today, it is a crucial technology in various applications, from autonomous vehicles to medical diagnostics and surveillance systems.

## **Implemented Models**

The following are the Computer Vision models I have implemented to date:

1. **[Image Classification](https://github.com/JersonGB22/ImageClassification-TensorFlow):** This task involves assigning a label or class to an image. Inputs are pixel values that compose an image, either in grayscale or RGB. Key use cases include medical image classification, social media photo categorization, and inventory product detection.

2. **[Object Detection](https://github.com/JersonGB22/ObjectDetection-TensorFlow-PyTorch):** These models identify and locate instances of objects such as cars, humans, buildings, animals, etc., in an image or video sequence. They return bounding box coordinates along with class labels. Current use cases include security surveillance, autonomous driving, and augmented reality.

3. **[Image Segmentation](https://github.com/JersonGB22/ImageClassification-TensorFlow):** This task classifies each pixel of an image into a specific category or instance, producing clearly defined areas for each class or object. It is divided into three main types:
   - **Semantic Segmentation:** Assigns a class label to each pixel without distinguishing between different instances of the same class.
   - **Instance Segmentation:** Labels each pixel and differentiates between individual instances.
   - **Panoptic Segmentation:** Combines both techniques for detailed segmentation.
   Use cases include medicine (tissue segmentation), agriculture (crop detection), and robotics.

4. **[Video Classification](https://github.com/JersonGB22/ImageClassification-TensorFlow):** This task involves assigning a label or class to a video. Models process video frames and generate the probability of each class being represented. Important use cases are activity detection in security, multimedia content classification, and sports analysis.

5. **[Image Captioning](https://github.com/JersonGB22/ImageClassification-TensorFlow):** This multimodal task combines Computer Vision and Natural Language Processing (NLP) to generate textual descriptions of images. These models are useful for describing images on social media platforms, improving accessibility for visually impaired individuals, and indexing images for search engines.

## **Contributions**

Contributions to this repository are welcome. If you have any questions or suggestions, please do not hesitate to contact me.

## **Technological Stack**
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)](https://docs.python.org/3/) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=101010)](https://www.tensorflow.org/api_docs)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=101010)](https://pytorch.org/docs/stable/index.html)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=101010)](https://huggingface.co/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-1572B6?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNTIgMjY0IiBoZWlnaHQ9IjI2NCIgd2lkdGg9IjI1MiI+CjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMF8xMDA4Xzk0MTc3KSI+CjxtYXNrIGhlaWdodD0iMjU4IiB3aWR0aD0iMjU0IiB5PSI2IiB4PSItMSIgbWFza1VuaXRzPSJ1c2VyU3BhY2VPblVzZSIgc3R5bGU9Im1hc2stdHlwZTpsdW1pbmFuY2UiIGlkPSJtYXNrMF8xMDA4Xzk0MTc3Ij4KPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yNTIuMzkxIDYuMDY3ODFILTAuNDM3NVYyNjMuOTMySDI1Mi4zOTFWNi4wNjc4MVoiPjwvcGF0aD4KPC9tYXNrPgo8ZyBtYXNrPSJ1cmwoI21hc2swXzEwMDhfOTQxNzcpIj4KPHBhdGggZmlsbD0iIzBCMjNBOSIgZD0iTTU4Ljc1IDYuMDY4NTFDMjYuMTEzNiA2LjA2ODUxIC0wLjQzNzUgMzIuNjMxOCAtMC40Mzc1IDY1LjI4MjlDLTAuNDM3NSA5Ny45MzE0IDI2LjExMzYgMTI0LjQ5NiA1OC43NSAxMjQuNDk2QzkxLjM4NyAxMjQuNDk2IDExNy45MzggOTcuOTMxNCAxMTcuOTM4IDY1LjI4MjlDMTE3LjkzOCAzMi42MzE4IDkxLjM4NyA2LjA2ODUxIDU4Ljc1IDYuMDY4NTFaIj48L3BhdGg+CjxwYXRoIGZpbGw9IiMwQjIzQTkiIGQ9Ik0xMjUuNzE5IDE5MS40ODlDMTA0LjM5OSAxOTEuNDg5IDg0LjI1NDIgMTg2LjA4OCA2Ni41NzAzIDE3Ni42MDZWMjAzLjQ3MUM2Ni41NzAzIDIzNi4wNzEgOTIuNTg5OSAyNjMuMTIxIDEyNS4xNzUgMjYzLjQzNkMxNTguMDc4IDI2My43NTQgMTg0Ljk0NyAyMzcuMDY5IDE4NC45NDcgMjA0LjIyNlYxNzYuNTgxQzE2Ny4yNDcgMTg2LjA4NyAxNDcuMDY5IDE5MS40ODkgMTI1LjcxOSAxOTEuNDg5WiI+PC9wYXRoPgo8cGF0aCBmaWxsPSJ1cmwoI3BhaW50MF9saW5lYXJfMTAwOF85NDE3NykiIGQ9Ik0xMzMuNDY2IDY1LjI4OTVDMTMzLjQwNSAxMDYuNDgxIDk5Ljk3OTYgMTM5LjkzNCA1OC42NTg0IDE0MC4wMzVDNDIuNzE4NSAxNDAuMDc2IDI3Ljc2MDcgMTM1LjExMiAxNS41NTQ3IDEyNi40NDVDMzcuMTg4OCAxNjUuMTM0IDc4LjQ3MDEgMTkxLjUxNCAxMjUuNjczIDE5MS40MjNDMTk0LjI0MyAxOTEuNDc3IDI1MC44MjMgMTM1LjYyNiAyNTEuOTY2IDY3LjEyNTNMMjUxLjgwNCA2Ni45Nzg3QzI1MS44NzEgNjUuMjcxOSAyNTEuNzg4IDY2LjY3IDI1MS44NzEgNjUuMjcxOUMyNTEuOTA0IDMyLjU5OCAyMjUuMzEyIDUuOTMxMDEgMTkyLjgxNSA2LjA0NDc3QzE2MC4wMDggNi4xNzQzMiAxMzMuNDk5IDMyLjYxNTIgMTMzLjQ2NiA2NS4yODk1WiI+PC9wYXRoPgo8L2c+CjwvZz4KPGRlZnM+CjxsaW5lYXJHcmFkaWVudCBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeTI9IjI3LjA1MjciIHgyPSIyMTcuNzA4IiB5MT0iMTg5LjMyOCIgeDE9IjcxLjE1NTIiIGlkPSJwYWludDBfbGluZWFyXzEwMDhfOTQxNzciPgo8c3RvcCBzdG9wLWNvbG9yPSIjMDlEQkYwIj48L3N0b3A+CjxzdG9wIHN0b3AtY29sb3I9IiMwQjIzQTkiIG9mZnNldD0iMSI+PC9zdG9wPgo8L2xpbmVhckdyYWRpZW50Pgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEwMDhfOTQxNzciPgo8cmVjdCBmaWxsPSJ3aGl0ZSIgaGVpZ2h0PSIyNjQiIHdpZHRoPSIyNTIiPjwvcmVjdD4KPC9jbGlwUGF0aD4KPC9kZWZzPgo8L3N2Zz4K&logoColor=white&labelColor=101010)](https://docs.ultralytics.com/)

[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=101010)](https://scikit-learn.org/stable/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white&labelColor=101010)](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white&labelColor=101010)](https://dev.mysql.com/doc/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white&labelColor=101010)](https://plotly.com/)

## **Contact**
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white&labelColor=101010)](mailto:jerson.gimenesbeltran@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=101010)](https://www.linkedin.com/in/jerson-gimenes-beltran/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white&labelColor=101010)](https://github.com/JersonGB22/)
