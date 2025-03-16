<h1 align="center">Awesome-MCoT</h1>
Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey

# 🎨 Introduction

Multimodal chain-of-thought (MCoT) reasoning has garnered attention for its ability to enhance ***step-by-step*** reasoning in multimodal contexts, particularly within multimodal large language models (MLLMs). Current MCoT research explores various methodologies to address the challenges posed by images, videos, speech, audio, 3D data, and structured data, achieving success in fields such as robotics, healthcare, and autonomous driving. However, despite these advancements, the field lacks a comprehensive review that addresses the numerous remaining challenges.

To fill this gap, we present **_the first systematic survey of MCoT reasoning_**, elucidating the foundational concepts and definitions pertinent to this area. Our work includes a detailed taxonomy and an analysis of existing methodologies across different applications, as well as insights into current challenges and future research directions aimed at fostering the development of multimodal reasoning.

<p align="center">
<!--   <img src="assets/mcot_tasks.png" width="75%"> -->
</p>

# 📕 Table of Contents

- [🌷 MCoT Datasets and Benchmarks](#-datasets)
- [🍕 Scene Graph Generation](#-scene-graph-generation)
  - [2D (Image) Scene Graph Generation](#2d-image-scene-graph-generation)
  - [Panoptic Scene Graph Generation](#panoptic-scene-graph-generation)

---

---

<!-- CVPR-8A2BE2 -->
<!-- WACV-6a5acd -->
<!-- NIPS-CD5C5C -->
<!-- ICML-FF7F50 -->
<!-- ICCV-00CED1 -->
<!-- ECCV-1e90ff -->
<!-- TPAMI-BC8F8F -->
<!-- IJCAI-228b22 -->
<!-- AAAI-c71585 -->
<!-- arXiv-b22222 -->
<!-- ACL-191970 -->
<!-- TPAMI-ffa07a -->


# MCoT Over Various Modalities
### MCoT Reasoning Over Image


### MCoT Reasoning Over Video


title - arxiv_html - paper_icon - github



### MCoT Reasoning Over Audio and Speech
+ [**CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought**](https://arxiv.org/abs/2409.19510) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/X-LANCE/SLAM-LLM.svg?style=social&label=Star)](https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/st_covost2)  
+ [**Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model**](https://arxiv.org/abs/2501.07246) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Leveraging Chain of Thought towards Empathetic Spoken Dialogue without Corresponding Question-Answering Data**](https://arxiv.org/abs/2501.10937) [![Paper](https://img.shields.io/badge/ICASSP-FFC0CB)]()  
+ [**Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models**](https://arxiv.org/abs/2503.02318) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation**](https://arxiv.org/abs/2410.10676) [![Paper](https://img.shields.io/badge/ICLR-8A2BE2)]() [![Star](https://img.shields.io/github/stars/PeiwenSun2000/Both-Ears-Wide-Open.svg?style=social&label=Star)](https://github.com/PeiwenSun2000/Both-Ears-Wide-Open)  
+ [**SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation**](https://arxiv.org/abs/2401.13527) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/0nutation/SpeechGPT.svg?style=social&label=Star)](https://github.com/0nutation/SpeechGPT)  

### MCoT Reasoning Over Table and Chart
+ [**LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding**](https://arxiv.org/pdf/2404.05225) [![Paper](https://img.shields.io/badge/CVPR-8A2BE2)]()  
+ [**Multimodal Graph Constrastive Learning and Prompt for ChartQA**](https://arxiv.org/abs/2501.04303) [![arXiv](https://img.shields.io/badge/arXiv-b22222)]()  -
+ [**TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT**](https://arxiv.org/abs/2307.08674) [![ACL](https://img.shields.io/badge/ACL-191970)]()  [![Star](https://img.shields.io/github/stars/microsoft/Table-GPT.svg?style=social&label=Star)](https://github.com/microsoft/Table-GPT) 
+ [**Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**](https://arxiv.org/abs/2401.04398) [![ICLR](https://img.shields.io/badge/ICLR-1a1a1a)]()  [![Star](https://img.shields.io/github/stars/google-research/chain-of-table.svg?style=social&label=Star)](https://github.com/google-research/chain-of-table) 
+ [**ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding**](https://arxiv.org/abs/2501.05452) [![arXiv](https://img.shields.io/badge/arXiv-b22222)]()  [![Star](https://img.shields.io/github/stars/zeyofu/ReFocus_Code.svg?style=social&label=Star)](https://github.com/zeyofu/ReFocus_Code) 

### Cross-modal CoT Reasoning
+ [**AVQA-CoT: When CoT Meets Question Answering in Audio-Visual Scenarios**](https://sightsound.org/papers/2024/Li_AVQA-CoT_When_CoT_Meets_Question_Answering_in_Audio-Visual_Scenarios.pdf) [![Paper](https://img.shields.io/badge/Workshop-228b22)]()  
+ [**Can Textual Semantics Mitigate Sounding Object Segmentation Preference?**](https://arxiv.org/abs/2407.10947) [![Paper](https://img.shields.io/badge/ECCV-1e90ff)]() [![Star](https://img.shields.io/github/stars/GeWu-Lab/Sounding-Object-Segmentation-Preference.svg?style=social&label=Star)](https://github.com/GeWu-Lab/Sounding-Object-Segmentation-Preference)  
+ [**Chain of Empathy: Enhancing Empathetic Response of Large Language Models Based on Psychotherapy Models**](https://arxiv.org/abs/2311.04915) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Multimodal PEAR Chain-of-Thought Reasoning for Multimodal Sentiment Analysis**](https://dl.acm.org/doi/10.1145/3672398) [![Paper](https://img.shields.io/badge/ACMMM-FFD700)]()  
+ [**R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning**](https://arxiv.org/abs/2503.05379) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/HumanMLLM/R1-Omni.svg?style=social&label=Star)](https://github.com/HumanMLLM/R1-Omni)  

---


# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yaotingwangofficial/Awesome-MCoT&type=Date)](https://star-history.com/#yaotingwangofficial/Awesome-MCoT&Date)

