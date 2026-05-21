# Contributing to Awesome-MCoT

Thanks for helping keep this survey of Multimodal Chain-of-Thought (MCoT) reasoning up to date. This guide describes how to add a paper, dataset, or benchmark so that all entries remain consistent and easy to scan.

If anything is unclear, open a [Discussion](https://github.com/yaotingwangofficial/Awesome-MCoT/discussions) before submitting a PR.

---

## Scope

We curate work that **substantively involves chain-of-thought (CoT) style reasoning over multimodal inputs or outputs**, including but not limited to:

- VLM / MLLM reasoning over images, video, 3D, audio, speech, tables, charts, or cross-modal combinations
- Datasets and benchmarks that explicitly evaluate or train MCoT (with or without rationales)
- Reinforcement learning, test-time scaling, or self-verification methods applied to multimodal reasoning
- Applications of MCoT in embodied AI, autonomous driving, healthcare, agents, generation, etc.

Out-of-scope: pure unimodal CoT (text-only) and generic VLM work without an explicit reasoning component.

---

## How to Submit

1. **Open an issue first** (recommended) using the *Paper suggestion* template if you are unsure about scope or placement.
2. **Fork** the repo and create a branch from `main`: `add/<short-paper-slug>`.
3. **Add your entry** to the appropriate section (see [Where Does My Paper Go?](#where-does-my-paper-go)).
4. **Open a Pull Request** filling in the PR template.

One PR per paper is preferred, unless the papers form an obvious cluster (e.g., a method + its dataset).

---

## Entry Format

### Paper entry (used in modality and application sections)

```markdown
- [**Paper Title**](paper-url) [![Paper](https://img.shields.io/badge/VENUE-COLOR)](paper-url) [![Star](https://img.shields.io/github/stars/USER/REPO.svg?style=social&label=Star)](https://github.com/USER/REPO) <!-- MM-YY -->
```

Rules:

- **Title** is wrapped in `[**...**]` (bold) and links to the canonical paper URL (prefer arXiv abstract page, then the official venue page, then a project page).
- **Venue badge** uses the color palette below. If the paper is preprint-only, use `arXiv-b22222`.
- **Code badge** is included only if a public repository exists.
- **Optional badges** (in this order, when applicable): HuggingFace, project page, dataset, demo.
- **Trailing comment** `<!-- MM-YY -->` records the first public release (arXiv v1 or venue date), used internally for sorting.

Example:

```markdown
- [**Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models**](https://arxiv.org/abs/2503.06749) [![Paper](https://img.shields.io/badge/arXiv-b22222)](https://arxiv.org/abs/2503.06749) [![Star](https://img.shields.io/github/stars/Osilly/Vision-R1.svg?style=social&label=Star)](https://github.com/Osilly/Vision-R1) <!-- 03-25 -->
```

### Dataset / benchmark entry (used in Tab-1/2/3)

A new row in the corresponding Markdown table. Match the existing column count and ordering exactly; keep cell content concise. If a field is unknown, use `-`.

### Reinforcement-learning method entry (RL section)

A new row in the RL table. Columns: `Name | Base Model | Modality | Method | Cold Start | RL Algo | Open Source`.

---

## Venue Color Palette

Use these existing badge colors so the page stays visually consistent. If you need a venue not on this list, pick a reasonable hex and add it here in the same PR.

| Venue | Badge | Color |
|---|---|---|
| arXiv | `arXiv-b22222` | `#b22222` |
| CVPR | `CVPR-8A2BE2` | `#8A2BE2` |
| ICCV | `ICCV-00CED1` | `#00CED1` |
| ECCV | `ECCV-1e90ff` | `#1e90ff` |
| WACV | `WACV-6a5acd` | `#6a5acd` |
| NeurIPS | `NIPS-CD5C5C` | `#CD5C5C` |
| ICML | `ICML-FF7F50` | `#FF7F50` |
| ICLR | `ICLR-355C7D` | `#355C7D` |
| AAAI | `AAAI-c71585` | `#c71585` |
| IJCAI | `IJCAI-228b22` | `#228b22` |
| TPAMI | `TPAMI-BC8F8F` | `#BC8F8F` |
| ACL | `ACL-191970` | `#191970` |
| EMNLP | `EMNLP-191970` | `#191970` |
| ACMMM | `ACMMM-FFD700` | `#FFD700` |

---

## Where Does My Paper Go?

| If the paper… | Add to |
|---|---|
| Introduces a *training* dataset with rationales | **Tab-1** |
| Introduces a *benchmark* (no rationales) | **Tab-2** |
| Introduces a *benchmark with rationales* | **Tab-3** |
| Uses RL to incentivize multimodal reasoning | **Multimodal Reasoning via RL** table |
| Focuses on a specific input modality | The matching subsection under **MCoT Over Various Modalities** |
| Proposes a general methodology (rationale construction, structural reasoning, test-time scaling, …) | The matching subsection under **MCoT Methodologies** |
| Applies MCoT to a domain (embodied / driving / medical / …) | The matching subsection under **Applications with MCoT Reasoning** |

Within each subsection, papers are grouped by **year (descending)** inside `<details>` blocks. Place your entry at the top of the year sub-block; sub-block ordering inside a year is loose (newer arXiv versions first is a good default).

---

## Quality Checklist (before opening a PR)

- [ ] Paper is in scope and not already listed (search the README first).
- [ ] Entry follows the exact format above; badges render correctly when previewing the PR.
- [ ] All links resolve (no empty `()` link targets).
- [ ] Year placement and `<!-- MM-YY -->` comment match the paper's first public release.
- [ ] No personal commentary in the entry; descriptions belong in the paper itself.

---

## Style Conventions

- Use `-` for list bullets (not `+` or `*`).
- Wrap paper titles with `**...**` (bold) inside the link text.
- Prefer arXiv abstract URLs (`/abs/xxxx.xxxxx`) over PDF URLs.
- Keep entries on a **single line**; do not break long titles.
- Do not change unrelated parts of the README in a paper PR.

---

## License & Attribution

By contributing, you agree that your contribution may be redistributed under the repository's existing license. If you are adding your own paper, that is welcomed and encouraged — please disclose it in the PR description.
