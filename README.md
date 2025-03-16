<h1 align="center">Awesome-MCoT</h1>
Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey

# 🎨 Introduction

Multimodal chain-of-thought (MCoT) reasoning has garnered attention for its ability to enhance ***step-by-step*** reasoning in multimodal contexts, particularly within multimodal large language models (MLLMs). Current MCoT research explores various methodologies to address the challenges posed by images, videos, speech, audio, 3D data, and structured data, achieving success in fields such as robotics, healthcare, and autonomous driving. However, despite these advancements, the field lacks a comprehensive review that addresses the numerous remaining challenges.

To fill this gap, we present **_the first systematic survey of MCoT reasoning_**, elucidating the foundational concepts and definitions pertinent to this area. Our work includes a detailed taxonomy and an analysis of existing methodologies across different applications, as well as insights into current challenges and future research directions aimed at fostering the development of multimodal reasoning.

<p align="center">
  <img src="assets/mcot_tasks.png" width="75%">
</p>

# 📕 Table of Contents

- [🌷 MCoT Datasets and Benchmarks](#-datasets)
- [🍕 Scene Graph Generation](#-scene-graph-generation)
  - [2D (Image) Scene Graph Generation](#2d-image-scene-graph-generation)
  - [Panoptic Scene Graph Generation](#panoptic-scene-graph-generation)

---

# MCoT Datasets and Benchmarks

| Datasets                  | Year | Task              | Domain               | Modality | Format         | Samples   |
|---------------------------|------|-------------------|----------------------|----------|----------------|-----------|
|                           |      |                   | **training with rationale**     |     |          |           |  
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
|                           |      |                   |                      |          |                |           |
|                           |      |                   |  **evaluation without rationale** |          |              |        
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
|                           |      |                   |                      |          |                |           |
|                           |      |                   |   **evaluation with rationale**  |          |                |        
| CoMT                      | 2024 | VQA               | Common               | T, I     | MC             | 3,853     |
| OmniBench                 | 2024 | VideoQA           | Common               | T, I, A  | MC             | 1,142     |
| WorldQA                   | 2024 | VideoQA           | Common               | T, V, A  | Open           | 1,007     |
| MiCEval                   | 2024 | VQA               | Common               | T, I     | Open           | 643       |
| OlympiadBench             | 2024 | ScienceQA         | Maths, Physics       | T, I     | Open           | 8,476     |
| MME-CoT                   | 2025 | VQA               | Science, Math, Common| T, I     | MC and Open    | 1,130     |
| EMMA                      | 2025 | VQA               | Science              | T, I     | MC and Open    | 2,788     |
| VisualProcessBench        | 2025 | ScienceQA         | Math, Science        | T, I     | MC and Open    | 2,866     |

**Caption:** Datasets and Benchmarks for MCoT Training and Evaluation. "MC" and "Open" refer to multiple-choice and open-ended answer formats, while "T", "I", "V", and "A" represent Text, Image, Video, and Audio, respectively.



