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
<!-- EMNLP-191970 -->
<!-- TPAMI-ffa07a -->
title - arxiv_html - paper_icon - github


# MCoT Over Various Modalities
### MCoT Reasoning Over Image
+ [**See, Think, Confirm: Interactive Prompting Between Vision and Language Models for Knowledge-based Visual Reasoning**](https://arxiv.org/abs/2301.05226) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() 
+ [**Multimodal Chain-of-Thought Reasoning in Language Models**](https://arxiv.org/abs/2302.00923) [![Paper](https://img.shields.io/badge/TMLR-FFA07A)]() [![Star](https://img.shields.io/github/stars/amazon-science/mm-cot.svg?style=social&label=Star)](https://github.com/amazon-science/mm-cot) 
+ [**MC-CoT: A Modular Collaborative CoT Framework for Zero-shot Medical-VQA with LLM and MLLM Integration**](https://arxiv.org/abs/2410.04521) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/thomaswei-cn/MC-CoT.svg?style=social&label=Star)](https://github.com/thomaswei-cn/MC-CoT) 
+ [**Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching**](https://arxiv.org/abs/2503.05179) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/SimonAytes/SoT.svg?style=social&label=Star)](https://github.com/SimonAytes/SoT) 
+ [**Compositional Chain-of-Thought Prompting for Large Multimodal Models**](https://arxiv.org/abs/2311.17076) [![Paper](https://img.shields.io/badge/CVPR-8A2BE2)]() [![Star](https://img.shields.io/github/stars/chancharikmitra/CCoT.svg?style=social&label=Star)](https://github.com/chancharikmitra/CCoT) 
+ [**CoCoT: Contrastive Chain-of-Thought Prompting for Large Multimodal Models with Multiple Image Inputs**](https://arxiv.org/abs/2401.02582) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() 
+ [**RelationLMM: Large Multimodal Model as Open and Versatile Visual Relationship Generalist**](https://ieeexplore.ieee.org/document/10845195) [![Paper](https://img.shields.io/badge/TPAMI-FFA07A)]() 
+ [**Thinking Like an Expert:Multimodal Hypergraph-of-Thought (HoT) Reasoning to boost Foundation Modals**](https://arxiv.org/abs/2308.06207) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() 
+ [**DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models**](https://arxiv.org/abs/2310.16436) [![Paper](https://img.shields.io/badge/NIPS-CD5C5C)]() [![Star](https://img.shields.io/github/stars/SooLab/DDCOT.svg?style=social&label=Star)](https://github.com/SooLab/DDCOT) 
+ [**The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models**](https://arxiv.org/abs/2305.14999) [![Paper](https://img.shields.io/badge/ACL-191970)]() [![Star](https://img.shields.io/github/stars/VT-NLP/SOCRATIC-QUESTIONING.svg?style=social&label=Star)](https://github.com/VT-NLP/SOCRATIC-QUESTIONING) 
+ [**Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models**](https://arxiv.org/abs/2403.12966) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/dongyh20/Chain-of-Spot.svg?style=social&label=Star)](https://github.com/dongyh20/Chain-of-Spot) 
+ [**TextCoT: Zoom In for Enhanced Multimodal Text-Rich Image Understanding**](https://arxiv.org/abs/2404.09797) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/bzluan/TextCoT.svg?style=social&label=Star)](https://github.com/bzluan/TextCoT) 
+ [**DCoT: Dual Chain-of-Thought Prompting for Large Multimodal Models**](https://openreview.net/forum?id=0saecDOdh2) [![Paper](https://img.shields.io/badge/ACML-808080)]() 
+ [**RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models**](https://arxiv.org/abs/2404.12065) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()
+ [**Cantor: Inspiring Multimodal Chain-of-Thought of MLLM**](https://arxiv.org/abs/2404.16033) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/ggg0919/cantor.svg?style=social&label=Star)](https://github.com/ggg0919/cantor)
+ [**KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts Reasoning**](https://arxiv.org/abs/2401.12863) [![Paper](https://img.shields.io/badge/AAAI-c71585)]()
+ [**PKRD-CoT: A Unified Chain-of-thought Prompting for Multi-Modal Large Language Models in Autonomous Driving**](https://arxiv.org/abs/2412.02025) [![Paper](https://img.shields.io/badge/ICONIP-2F4F4F)]()
+ [**Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models**](https://arxiv.org/abs/2305.16582) [![Paper](https://img.shields.io/badge/ACL-191970)]() [![Star](https://img.shields.io/github/stars/Zoeyyao27/Graph-of-Thought.svg?style=social&label=Star)](https://github.com/Zoeyyao27/Graph-of-Thought)
+ [**A Picture Is Worth a Graph: A Blueprint Debate Paradigm for Multimodal Reasoning**](https://arxiv.org/abs/2403.14972) [![Paper](https://img.shields.io/badge/ACMMM-FFFF00)]() [![Star](https://img.shields.io/github/stars/thecharm/BDoG.svg?style=social&label=Star)](https://github.com/thecharm/BDoG)
+ [**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/abs/2312.00438) [![Paper](https://img.shields.io/badge/ECCV-1e90ff)]() [![Star](https://img.shields.io/github/stars/vlm-driver/Dolphins.svg?style=social&label=Star)](https://github.com/vlm-driver/Dolphins)
+ [**Enhancing Large Vision Language Models with Self-Training on Image Comprehension**](https://arxiv.org/abs/2405.19716) [![Paper](https://img.shields.io/badge/NIPS-CD5C5C)]() [![Star](https://img.shields.io/github/stars/yihedeng9/STIC.svg?style=social&label=Star)](https://github.com/yihedeng9/STIC)
+ [**PS-CoT-Adapter: adapting plan-and-solve chain-of-thought for ScienceQA**](https://link.springer.com/article/10.1007/s11432-024-4211-9) [![Paper](https://img.shields.io/badge/SCIC-2F4F4F)]() [![Star](https://img.shields.io/github/stars/Sunhxxin/PS-CoT-Adapter.svg?style=social&label=Star)](https://github.com/Sunhxxin/PS-CoT-Adapter)
+ [**Enhancing Semantics in Multimodal Chain of Thought via Soft Negative Sampling**](https://aclanthology.org/2024.lrec-main.537/) [![Paper](https://img.shields.io/badge/COLING-191970)]()
+ [**Chain-of-Exemplar: Enhancing Distractor Generation for Multimodal Educational Question Generation**](https://aclanthology.org/2024.acl-long.432/) [![Paper](https://img.shields.io/badge/ACL-191970)]() [![Star](https://img.shields.io/github/stars/Luohh5/Chain-of-Exemplar.svg?style=social&label=Star)](https://github.com/Luohh5/Chain-of-Exemplar)
+ [**R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models**](https://arxiv.org/abs/2410.17885) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/dle666/r-cot.svg?style=social&label=Star)](https://github.com/dle666/r-cot)
+ [**Multi-modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models**](https://ojs.aaai.org/index.php/AAAI/article/view/29776/31338) [![Paper](https://img.shields.io/badge/AAAI-c71585)]() [![Star](https://img.shields.io/github/stars/shimurenhlq/DPMM-COT.svg?style=social&label=Star)](https://github.com/shimurenhlq/DPMM-COT)
+ [**Perception Tokens Enhance Visual Reasoning in Multimodal Language Models**](https://arxiv.org/abs/2412.03548v2) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/mahtabbigverdi/Aurora.svg?style=social&label=Star)](https://github.com/mahtabbigverdi/Aurora/tree/main)
+ [**Visual CoT: Advancing MLLMs with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning**](https://arxiv.org/abs/2403.16999) [![Paper](https://img.shields.io/badge/NIPS-CD5C5C)]() [![Star](https://img.shields.io/github/stars/deepcs233/Visual-CoT.svg?style=social&label=Star)](https://github.com/deepcs233/Visual-CoT)


title - arxiv_html - paper_icon - github
+ RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models - https://arxiv.org/abs/2404.12065 -
+ Cantor: Inspiring Multimodal Chain-of-Thought of MLLM - https://arxiv.org/abs/2404.16033 - https://github.com/ggg0919/cantor
+ KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts Reasoning - https://arxiv.org/abs/2401.12863 - AAAI -
+ PKRD-CoT: A Unified Chain-of-thought Prompting for Multi-Modal Large Language Models in Autonomous Driving - https://arxiv.org/abs/2412.02025 - ICONIP -
+ Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models - https://arxiv.org/abs/2305.16582 - acl - https://github.com/Zoeyyao27/Graph-of-Thought
+ A Picture Is Worth a Graph: A Blueprint Debate Paradigm for Multimodal Reasoning - https://arxiv.org/abs/2403.14972 - acmmm - https://github.com/thecharm/BDoG
+ Dolphins: Multimodal Language Model for Driving - https://arxiv.org/abs/2312.00438 - eccv - https://github.com/vlm-driver/Dolphins
+ Enhancing Large Vision Language Models with Self-Training on Image Comprehension - https://arxiv.org/abs/2405.19716 - nips - https://github.com/yihedeng9/STIC
+ PS-CoT-Adapter: adapting plan-and-solve chain-of-thought for ScienceQA - https://link.springer.com/article/10.1007/s11432-024-4211-9 - SCIC - https://github.com/Sunhxxin/PS-CoT-Adapter
+ Enhancing Semantics in Multimodal Chain of Thought via Soft Negative Sampling - https://aclanthology.org/2024.lrec-main.537/ - COLING
+ Chain-of-Exemplar: Enhancing Distractor Generation for Multimodal Educational Question Generation - https://aclanthology.org/2024.acl-long.432/ - acl - https://github.com/Luohh5/Chain-of-Exemplar
+ R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models - https://arxiv.org/abs/2410.17885 - arxiv - https://github.com/dle666/r-cot
+ Multi-modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models - https://ojs.aaai.org/index.php/AAAI/article/view/29776/31338 - aaai - https://github.com/shimurenhlq/DPMM-COT
+ Perception Tokens Enhance Visual Reasoning in Multimodal Language Models - https://arxiv.org/abs/2412.03548v2 - https://github.com/mahtabbigverdi/Aurora/tree/main
+ Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning - https://arxiv.org/abs/2403.16999 - nips - https://github.com/deepcs233/Visual-CoT
+ 
  




### MCoT Reasoning Over Video
+ [**IntentQA: Context-aware Video Intent Reasoning**](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_IntentQA_Context-aware_Video_Intent_Reasoning_ICCV_2023_paper.pdf) [![Paper](https://img.shields.io/badge/ICCV-00CED1)]() [![Star](https://img.shields.io/github/stars/JoseponLee/IntentQA.svg?style=social&label=Star)](https://github.com/JoseponLee/IntentQA)  
+ [**Hallucination Mitigation Prompts Long-term Video Understanding**](https://arxiv.org/abs/2406.11333) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/lntzm/CVPR24Track-LongVideo.svg?style=social&label=Star)](https://github.com/lntzm/CVPR24Track-LongVideo)  
+ [**AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?**](https://arxiv.org/abs/2307.16368) [![Paper](https://img.shields.io/badge/ICLR-8A2BE2)]() [![Star](https://img.shields.io/github/stars/brown-palm/AntGPT.svg?style=social&label=Star)](https://github.com/brown-palm/AntGPT)  
+ [**DreamFactory: Pioneering Multi-Scene Long Video Generation with a Multi-Agent Framework**](https://arxiv.org/abs/2408.11788) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition**](https://arxiv.org/abs/2501.03230) [![Paper](https://img.shields.io/badge/ICML-FF7F50)]() [![Star](https://img.shields.io/github/stars/scofield7419/Video-of-Thought.svg?style=social&label=Star)](https://github.com/scofield7419/Video-of-Thought)  
+ [**CaRDiff: Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion**](https://arxiv.org/abs/2408.12009) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning**](https://openreview.net/pdf/6c509d93a31887cb5e3feaae2a453c392028dfdb.pdf) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Let's Think Frame by Frame with VIP: A Video Infilling and Prediction Dataset for Evaluating Video Chain-of-Thought**](https://arxiv.org/abs/2305.13903) [![Paper](https://img.shields.io/badge/EMNLP-191970)]() [![Star](https://img.shields.io/github/stars/vaishnaviHimakunthala/VIP.svg?style=social&label=Star)](https://github.com/vaishnaviHimakunthala/VIP)  
+ [**CoS: Chain-of-Shot Prompting for Long Video Understanding**](https://arxiv.org/abs/2502.06428) [![Paper](https://img.shields.io/badge/arXiv-b22222)]() [![Star](https://img.shields.io/github/stars/lwpyh/CoS_codes.svg?style=social&label=Star)](https://github.com/lwpyh/CoS_codes)  
+ [**TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos**](https://arxiv.org/abs/2411.02570) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Interpretable Video based Stress Detection with Self-Refine Chain-of-thought Reasoning**](https://arxiv.org/abs/2410.09449) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Analyzing Key Factors Influencing Emotion Prediction Performance of VLLMs in Conversational Contexts**](https://aclanthology.org/2024.emnlp-main.331.pdf) [![Paper](https://img.shields.io/badge/EMNLP-191970)]()  
+ [**Large Vision-Language Models as Emotion Recognizers in Context Awareness**](https://arxiv.org/abs/2407.11300) [![Paper](https://img.shields.io/badge/ACML-808080)]() [![Star](https://img.shields.io/github/stars/ydk122024/CAER.svg?style=social&label=Star)](https://github.com/ydk122024/CAER)  


### MCoT Reasoning Over 3D
+ [**3D-PreMise: Can Large Language Models Generate 3D Shapes with Sharp Features and Parametric Control?**](https://arxiv.org/abs/2401.06437) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**L3GO: Language Agents with Chain-of-3D-Thoughts for Generating Unconventional Objects**](https://arxiv.org/abs/2402.09052) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models**](https://arxiv.org/abs/2310.18308) [![Paper](https://img.shields.io/badge/ICRA-000080)]() [![Star](https://img.shields.io/github/stars/pushkalkatara/Gen2Sim.svg?style=social&label=Star)](https://github.com/pushkalkatara/Gen2Sim)  
+ [**CoT3DRef: Chain-of-Thoughts Data-Efficient 3D Visual Grounding**](https://arxiv.org/abs/2310.06214) [![Paper](https://img.shields.io/badge/ICLR-8A2BE2)]() [![Star](https://img.shields.io/github/stars/eslambakr/CoT3D_VG.svg?style=social&label=Star)](https://github.com/eslambakr/CoT3D_VG)  
+ [**Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning**](https://arxiv.org/abs/2503.06232) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  

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

