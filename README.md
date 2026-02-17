# AI Training Materials for CodeTrain Africa

A comprehensive training curriculum for machine learning and deep learning.

## About This Project

This repository contains hands-on training materials designed to take students from Python fundamentals through advanced deep learning concepts. The curriculum is built using [Quarto](https://quarto.org/) to generate interactive notebooks and HTML documentation.

### Partners & Sponsors

- **Created by:** Speedykom Group GmbH in collaboration with WINS Global Consult GmbH
- **For:** CodeTrain Africa, Ghana
- **Sponsored by:** Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ) GmbH

## Learning Path

The materials are organized into progressive modules:

### 00 - Welcome
Project configuration and setup files.

### 01 - Setup
Environment setup guides and installation instructions to get students started with the required tools (Python, Jupyter, required packages).

### 02 - Version Control
Introduction to Git and GitHub for collaborative coding and version management.

### 03 - Python Basics
Python programming fundamentals covering:
- Variables, data types, and operators
- Control flow (loops, conditionals)
- Functions and modules
- Object-oriented programming

### 04 - Data Analysis with Pandas
Data manipulation and analysis using Pandas:
- DataFrames and Series
- Data cleaning and preprocessing
- Filtering, grouping, and aggregation
- Working with CSV and other data formats
- **Data folder:** `data/processed_startups.csv`

### 05 - Machine Learning
Introduction to machine learning with scikit-learn:
- Supervised learning (regression, classification)
- Model training and evaluation
- Feature engineering and preprocessing
- Cross-validation and hyperparameter tuning

### 06 - Deep Learning
Deep learning with PyTorch:
- Neural network fundamentals
- Convolutional Neural Networks (CNNs)
- Training loops and optimization
- Working with image data
- **Datasets:** MNIST, GTSRB (traffic signs), Cassava disease classification
- **Model:** Pre-trained cassava disease classifier (`cassava_model_full.pth`)

### 07 - Agentic AI
Introduction to agentic AI concepts and building AI agents with Opencode.

### 08 - Resources
Additional learning materials:
- **books/** - Reference textbooks (PDFs - not included in repository)
- **diagrams/** - Visual aids and diagrams
- **data/** - General datasets for exercises

## Getting Started

1. Install Python 3.x and required packages (see `01-setup/`)
2. Clone this repository
3. Navigate to the first module: `03-python-basics/`
4. Open the `.ipynb` (Jupyter notebook) or `.html` files to start learning

## File Types Explained

- **.qmd** - Quarto source files (markdown + code)
- **.ipynb** - Jupyter notebooks (interactive Python)
- **.html** - Rendered HTML pages (readable in any browser)
- **.quarto_ipynb** - Quarto-generated notebooks

## Technical Requirements

- Python 3.8+
- Jupyter Lab or Jupyter Notebook
- Required packages (see `pyproject.toml`)
- Quarto (for generating materials)

## License

These materials are provided for educational purposes to CodeTrain Africa students.

---

**Questions?** Contact your instructor or refer to the guides in `01-setup/` and `02-version-control/`.
