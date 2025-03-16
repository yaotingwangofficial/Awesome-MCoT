<h1 align="center">Awesome-MCoT</h1>
<h2 align="center">Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey</h2>


# 🎇 Introduction

Multimodal chain-of-thought (MCoT) reasoning has garnered attention for its ability to enhance ***step-by-step*** reasoning in multimodal contexts, particularly within multimodal large language models (MLLMs). Current MCoT research explores various methodologies to address the challenges posed by images, videos, speech, audio, 3D data, and structured data, achieving success in fields such as robotics, healthcare, and autonomous driving. However, despite these advancements, the field lacks a comprehensive review that addresses the numerous remaining challenges.

To fill this gap, we present **_the first systematic survey of MCoT reasoning_**, elucidating the foundational concepts and definitions pertinent to this area. Our work includes a detailed taxonomy and an analysis of existing methodologies across different applications, as well as insights into current challenges and future research directions aimed at fostering the development of multimodal reasoning.

<p align="center">
  <img src="assets/mcot_tasks.png" width="60%">
</p>

---
# 📢 Updates
> 2025-03-17: 

---
# ⌛️ TODO
> 
---

# 📕 Table of Contents

- [🎖 MCoT Datasets and Benchmarks](#-mcot-datasets-and-benchmarks)
  - [Training with rationle](#tab-1-datasets-for-mcot-training-with-rationale)
  - [Evaluation without rationle](#tab-2-benchmarks-for-mcot-evaluation-without-rationale)
  - [Evaluation with rationle](#tab-3-benchmarks-for-mcot-evaluation-with-rationale)
- [✨ MCoT Over Various Modalities](#-mcot-over-various-modalities)
  - [MCoT Reasoning Over Image](#mcot-reasoning-over-image)
  - [MCoT Reasoning Over Video](#mcot-reasoning-over-video) 
  - [MCoT Reasoning Over 3D](#mcot-reasoning-over-3d) 
  - [MCoT Reasoning Over Audio and Speech](#mcot-reasoning-over-audio-and-speech) 
  - [MCoT Reasoning Over Table and Chart](#mcot-reasoning-over-table-and-chart) 
  - [Cross-modal CoT Reasoning](#cross-modal-cot-reasoning) 
- [🔥 Multimodal Reasoning Models](#-multimodal-reasoning-models)
  - [Image](#image)
  - [Video](#video)
  - [Audio](#audio)
  - [Omni](#omni)
- [🪄 MCoT Methodologies](#-mcot-methodologies)
  - [Rationale Construction](#rationale-construction)
  - [Structural Reasoning](#structural-reasoning)
  - [Information Enhancing](#information-enhancing)
  - [Objective Granularity](#objective-granularity)
  - [Multimodal Rationale](#multimodal-rationale)
  - [Test-time Scaling](#test-time-scaling)
- [🚀 Useful Links](#-useful-links)
- [🔦 Discussion Group](#-discussion-group)
- [⭐️ Star History](#%EF%B8%8F-star-history)

---

# 🎖 MCoT Datasets and Benchmarks
* "MC" and "Open" refer to multiple-choice and open-ended answer formats.
* "T", "I", "V", and "A" represent Text, Image, Video, and Audio, respectively.

### Tab-1: Datasets for MCoT _Training_ _with_ Rationale. 
| Datasets                  | Year | Task              | Domain               | Modality | Format         | Samples   |
|:---------------------------:|:------:|:-------------------:|:----------------------:|:----------:|:----------------:|:-----------:|
| ScienceQA                 | 2022 | VQA               | Science              | T, I     | MC             | 21K       |
| A-OKVQA                   | 2022 | VQA               | Common               | T, I     | MC             | 25K       |
| EgoCoT                    | 2023 | VideoQA           | Common               | T, V     | Open           | 200M      |
| VideoCoT                  | 2024 | VideoQA           | Human Action         | T, V     | Open           | 22K       |
| VideoEspresso             | 2024 | VideoQA           | Common               | T, V     | Open           | 202,164   |
| EMMA-X                    | 2024 | Robot Manipulation| Indoor               | T, V     | Robot Actions  | 60K       |
| M3CoT                     | 2024 | VQA               | Science, Math, Common| T, I     | MC             | 11.4K     |
| MAVIS                     | 2024 | ScienceQA         | Math                 | T, I     | MC and Open    | 834K      |
| LLaVA-CoT-100k            | 2024 | VQA               | Common, Science      | T, I     | MC and Open    | 834K      |
| MAmmoTH-VL                | 2024 | Diverse           | Diverse              | T, I     | MC and Open    | 12M       |
| Mulberry-260k             | 2024 | Diverse           | Diverse              | T, I     | MC and Open    | 260K      |
| MM-Verify                 | 2025 | MathQA            | Math                 | T, I     | MC and Open    | 59,772    |
| VisualPRM400K             | 2025 | ScienceQA         | Math, Science        | T, I     | MC and Open    | 400K      |
| R1-OneVision              | 2025 | Diverse           | Diverse              | T, I     | MC and Open    | 155K      |

### Tab-2: Benchmarks for MCoT _Evaluation_ _without_ Rationale.
| Datasets                  | Year | Task              | Domain               | Modality | Format         | Samples   |
|:---------------------------:|:------:|:-------------------:|:----------------------:|:----------:|:----------------:|:-----------:|
| MMMU                      | 2023 | VQA               | Arts, Science        | T, I     | MC and Open    | 11.5K     |
| SEED                      | 2023 | VQA               | Common               | T, I     | MC             | 19K       |
| MathVista                 | 2023 | ScienceQA         | Math                 | T, I     | MC and Open    | 6,141     |
| MathVerse                 | 2024 | ScienceQA         | Math                 | T, I     | MC and Open    | 15K       |
| Math-Vision               | 2024 | ScienceQA         | Math                 | T, I     | MC and Open    | 3040      |
| MeViS                     | 2023 | Referring VOS     | Common               | T, V     | Dense Mask     | 2K        |
| VSIBench                  | 2024 | VideoQA           | Indoor               | T, V     | MC and Open    | 5K        |
| HallusionBench            | 2024 | VQA               | Common               | T, I     | Yes-No         | 1,129     |
| AV-Odyssey                | 2024 | AVQA              | Common               | T, V, A  | MC             | 4,555     |
| AVHBench                  | 2024 | AVQA              | Common               | T, V, A  | Open           | 5,816     |
| RefAVS-Bench              | 2024 | Referring AVS     | Common               | T, V, A  | Dense Mask     | 4,770     |
| MMAU                      | 2024 | AQA               | Common               | T, A     | MC             | 10K       |
| AVTrustBench              | 2025 | AVQA              | Common               | T, V, A  | MC and Open    | 600K      |
| MIG-Bench                 | 2025 | Multi-image Grounding | Common          | T, I     | BBox           | 5.89K     |
| MedAgentsBench            | 2025 | MedicalQA         | Medical              | T, I     | MC and Open    | 862       |


### Tab-3: Benchmarks for MCoT _Evaluation_ _with_ Rationale.
| Datasets                  | Year | Task              | Domain               | Modality | Format         | Samples   |
|:---------------------------:|:------:|:-------------------:|:----------------------:|:----------:|:----------------:|:-----------:|
| CoMT                      | 2024 | VQA               | Common               | T, I     | MC             | 3,853     |
| OmniBench                 | 2024 | VideoQA           | Common               | T, I, A  | MC             | 1,142     |
| WorldQA                   | 2024 | VideoQA           | Common               | T, V, A  | Open           | 1,007     |
| MiCEval                   | 2024 | VQA               | Common               | T, I     | Open           | 643       |
| OlympiadBench             | 2024 | ScienceQA         | Maths, Physics       | T, I     | Open           | 8,476     |
| MME-CoT                   | 2025 | VQA               | Science, Math, Common| T, I     | MC and Open    | 1,130     |
| EMMA                      | 2025 | VQA               | Science              | T, I     | MC and Open    | 2,788     |
| VisualProcessBench        | 2025 | ScienceQA         | Math, Science        | T, I     | MC and Open    | 2,866     |

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


# ✨ MCoT Over Various Modalities
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
+ [**Chain of Images for Intuitively Reasoning**](https://arxiv.org/abs/2311.09241) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  [![Star](https://img.shields.io/github/stars/GraphPKU/CoI.svg?style=social&label=Star)](https://github.com/GraphPKU/CoI)
+ [**Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models**](https://arxiv.org/abs/2406.09403) [![Paper](https://img.shields.io/badge/NIPS-CD5C5C)]()  [![Star](https://img.shields.io/github/stars/Yushi-Hu/VisualSketchpad.svg?style=social&label=Star)](https://github.com/Yushi-Hu/VisualSketchpad)
+ [**Imagine while Reasoning in Space: Multimodal Visualization-of-Thought**](https://arxiv.org/abs/2501.07542) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  
+ [**Mind's Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models**](https://arxiv.org/abs/2404.03622) [![Paper](https://img.shields.io/badge/NIPS-CD5C5C)]()  [![Star](https://img.shields.io/github/stars/sitloboi2012/Visualization-of-Thought.svg?style=social&label=Star)](https://github.com/sitloboi2012/Visualization-of-Thought)
+ [**CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection**](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_CoTDet_Affordance_Knowledge_Prompting_for_Task_Driven_Object_Detection_ICCV_2023_paper.pdf) [![Paper](https://img.shields.io/badge/ICCV-00CED1)]()  [![Star](https://img.shields.io/github/stars/toneyaya/cotdet.svg?style=social&label=Star)](https://toneyaya.github.io/cotdet)
+ [**DetToolChain: A New Prompting Paradigm to Unleash Detection Ability of MLLM**](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_10) [![Paper](https://img.shields.io/badge/ECCV-1e90ff)]()  [![Star](https://img.shields.io/github/stars/yixuan730/DetToolChain.svg?style=social&label=Star)](https://github.com/yixuan730/DetToolChain)
+ [**CPSeg: Finer-grained Image Semantic Segmentation via Chain-of-Thought Language Prompting**](https://arxiv.org/abs/2310.16069) [![Paper](https://img.shields.io/badge/WACV-6a5acd)]()  
+ [**PromptCoT: Align Prompt Distribution via Adapted Chain-of-Thought**](https://ieeexplore.ieee.org/iel8/10654794/10654797/10656469.pdf) [![Paper](https://img.shields.io/badge/CVPR-8A2BE2)]()  
+ [**Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step**](https://arxiv.org/abs/2501.13926) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  [![Star](https://img.shields.io/github/stars/ZiyuGuo99/Image-Generation-CoT.svg?style=social&label=Star)](https://github.com/ZiyuGuo99/Image-Generation-CoT)
+ [**LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation**](https://arxiv.org/abs/2308.05095) [![Paper](https://img.shields.io/badge/ACMMM-yellow)]()  [![Star](https://img.shields.io/github/stars/LayoutLLM-T2I/LayoutLLM-T2I.svg?style=social&label=Star)](https://github.com/LayoutLLM-T2I/LayoutLLM-T2I)
+ [**CreatiLayout: Siamese Multimodal Diffusion Transformer for Creative Layout-to-Image Generation**](https://arxiv.org/abs/2412.03859) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  [![Star](https://img.shields.io/github/stars/HuiZhang0812/CreatiLayout.svg?style=social&label=Star)](https://github.com/HuiZhang0812/CreatiLayout)

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

# 🔥 Multimodal Reasoning Models
### Image

### Video
+ [**Video-R1: Towards Super Reasoning Ability in Video Understanding**](https://github.com/tulerfeng/Video-R1) [![Star](https://img.shields.io/github/stars/tulerfeng/Video-R1.svg?style=social&label=Star)](https://github.com/tulerfeng/Video-R1)

  
### Audio
+ [**Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models**](https://arxiv.org/abs/2503.02318) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  

### Omni
+ [**R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning**](https://arxiv.org/abs/2503.05379) [![Paper](https://img.shields.io/badge/arXiv-b22222)]()  [![Star](https://img.shields.io/github/stars/HumanMLLM/R1-Omni.svg?style=social&label=Star)](https://github.com/HumanMLLM/R1-Omni)

---
# 🪄 MCoT Methodologies
## Rationale Construction
## Structural Reasoning
## Information Enhancing
## Objective Granularity
## Multimodal Rationale
## Test-time Scaling


---
# 🚀 Useful Links
### Survey
+ [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/abs/2502.17419)
+ [Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning](https://arxiv.org/abs/2501.15602)
+ [Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models](https://arxiv.org/abs/2503.09567)

--- 
# 🔦 Discussion Group
> Check issue-02:
---

<!--title - arxiv_html - paper_icon - github-->
---
# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yaotingwangofficial/Awesome-MCoT&type=Date)](https://star-history.com/#yaotingwangofficial/Awesome-MCoT&Date)


