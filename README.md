# Ghana AI Talent Accelerator - Training Materials

A comprehensive 16-week curriculum transforming MERN developers into Python/AI engineers, with structured phases, milestone assessments, and capstone projects for 65+ students.

## About This Project

This repository contains pedagogical training materials with interactive notebooks, presentations, and practical exercises. Each module includes clear learning objectives, "Why" explanations of key concepts, and hands-on code examples.

### Partners & Sponsors

- **Created by:** Speedykom Group GmbH
- **For:** CodeTrain Africa, Ghana
- **Sponsored by:** Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ) GmbH

---

## 16-Week Program Overview (4 Phases)

| Phase | Weeks | Modules | Focus | Milestone |
|-------|-------|---------|-------|-----------|
| **1: Foundations** | 1-4 | 00, 01, 02, 03 | Python Basics | Mini Demo Day 1: Logic Script |
| **2: Analytics** | 5-8 | 04 | Data Analysis + Pandas | Mini Demo Day 2: EDA Report |
| **3: Intelligence** | 9-12 | 05 | Machine Learning | Mini Demo Day 3: ML Model |
| **4: Advanced AI** | 13-16 | 06, 07 | Deep Learning + Agents | Final Demo Day: Capstone |

**Total:** ~95 contact/study hours across 16 weeks (~6 hrs/week)

---

## Module Breakdown

### 00 - Welcome
Introduction and course overview. Key resources:
- `curriculum-pacing-guide.pdf` - Week-by-week breakdown, learning gates, and success metrics
- `learning-objectives-quick-reference.pdf` - Complete learning objectives for all 19 notebooks
- `python-vs-other-languages.pdf` - Why Python dominates AI/Data Science

### 01 - Setup
Environment setup for Python development:
- Installing Python with `uv`
- VS Code configuration
- Creating virtual environments
- Managing dependencies with `pyproject.toml`

### 02 - Version Control
Professional Git and GitHub workflow:
- Local and remote repositories
- Branching and merging
- Collaborative development
- Portfolio building

### 03 - Python Basics (3 Notebooks)
**Critical Gateway - Must Master Before Pandas**
- **Notebook 1:** Variables, data structures (lists, tuples, dicts, sets), indexing
- **Notebook 2:** Functions, scope, docstrings, Pythonic patterns
- **Notebook 3:** Comprehensions, generators, iterators, functional programming

### 04 - Data Analysis with Pandas (3 Notebooks)
**Critical Gateway - Foundation for ML**
- **Notebook 1:** DataFrames, Series, loading data, exploring data
- **Notebook 2:** Data transformation, cleaning, visualization, storytelling
- **Notebook 3:** Groupby, aggregation, merging, reshaping (Tidy Data)

Includes visual explanations: "Why DataFrames?", "Why Tidy Data?", "Why Matplotlib?", "Why Seaborn?"

### 05 - Machine Learning (3 Notebooks)
ML concepts and implementations:
- **Notebook 1:** ML workflow, train/test split, Linear Regression
- **Notebook 2:** Classification, decision boundaries, evaluation metrics
- **Notebook 3:** Clustering, dimensionality reduction, model selection

### 06 - Deep Learning (3 Notebooks)
Neural networks and PyTorch:
- **Notebook 1:** Neural network fundamentals, activation functions, backpropagation, optimizers
- **Notebook 2:** PyTorch implementation, CNNs, training loops, Dropout, Data Augmentation
- **Notebook 3:** Transfer learning, advanced architectures, model deployment

Includes "Why" explanations: activation functions, dropout, batch normalization, learning rate scheduling

### 07 - Agentic AI
Building AI agents with orchestration frameworks:
- LLMs and pattern matching at scale
- Retrieval Augmented Generation (RAG)
- Orchestration frameworks and job patterns
- Building interactive agents (chatbots, researchers, analyzers)

---

## Repository Contents

This repository contains **only these file types** (others are kept locally and built from source):

- **`.ipynb`** - Jupyter notebooks (executable, rendered from .qmd source)
- **`.pdf`** - PDF documents (curriculum guides, learning objectives, presentations)
- **`.pptx`** - PowerPoint presentations for instructors
- **`.py`** - Python utility scripts and helper functions
- **`.csv`** - Sample datasets for exercises
- **`.md`** - Documentation (README, guides)

---

## Getting Started

### For Students:
1. Clone this repository
2. Install Python 3.8+ and Jupyter
3. Start with Module 03 (`03-python-basics/intro_python1.ipynb`)
4. Follow the learning objectives at the top of each notebook
5. Complete the self-study exercises between classes

### For Instructors:
1. Review the `curriculum-pacing-guide.pdf` for week-by-week structure
2. Use `learning-objectives-quick-reference.pdf` to assess student progress
3. Open `.pptx` presentations to guide in-class discussions
4. Refer to the "Why" sections to explain motivation behind concepts

---

## Learning Approach

Each notebook includes:
- **Learning Objectives** - Clear outcomes students will achieve
- **Why Section** - Historical problem solved by the concept
- **How Section** - Implementation and practical examples
- **Exercises** - Hands-on practice

### Pedagogical Philosophy:
**Why (historical problem) → How (implementation) → Why Next (evolution)**

This bridges theory and practice, showing students not just *what* to code, but *why* the concept exists.

---

## Critical Learning Gates

**Gate 1 (Week 5):** Python Mastery
- ✅ Students write list comprehensions naturally
- ✅ Function scope is automatic, not confusing
- ✅ Code reads as Pythonic, not JavaScript-in-Python

**Gate 2 (Week 8):** Pandas Fluency
- ✅ `.loc[]` vs `.iloc[]` is automatic
- ✅ Groupby and aggregation are intuitive
- ✅ Data questions answered without syntax lookup

**Gate 3 (Week 12):** Capstone Project
- ✅ Working AI application (deep learning or agent)
- ✅ Real data + real model + real deployment/demo
- ✅ Technical documentation complete
- ✅ Portfolio-ready to show employers

---

## Technical Requirements

- **Python:** 3.8+
- **Jupyter:** JupyterLab or JupyterNotebook
- **Tools:** uv (Python package manager)
- **Deep Learning:** Google Colab GPU (free)

---

## File Organization

```
00-welcome/
  ├── curriculum-pacing-guide.pdf
  ├── learning-objectives-quick-reference.pdf
  └── python-vs-other-languages.pdf

03-python-basics/
  ├── intro_python1.ipynb
  ├── intro_python2.ipynb
  ├── intro_python3.ipynb
  └── python_basics_presentation.pptx

04-data-analysis/
  ├── intro_pandas1.ipynb
  ├── intro_pandas2.ipynb
  ├── intro_pandas3.ipynb
  └── data_analysis_presentation.pptx

[... similar structure for Modules 05-07 ...]
```

---

## License & Attribution

These materials are provided for educational purposes to CodeTrain Africa students. Created by Speedykom Group GmbH, sponsored by GIZ.

---

**For questions or feedback:** Contact your instructor or refer to the guides in each module folder.
