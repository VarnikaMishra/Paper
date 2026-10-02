# Hybrid Deep Learning Model for Software Defect Classification

Implementation aligned with the research paper:
> **"Hybrid Deep Learning Model for Software Defect Classification Using Code2Vec and LinkNet-BiLSTM Score Level Fusion"**  
> *Srinivasa Rao Katragadda & Sirisha Potluri, IJECE Journal (2025)*

---

## 🌟 Key Features & Pipeline

1. **Code2Vec AST Feature Extraction (`code2vec_extractor.py`)**:
   - Parses source code into Abstract Syntax Trees (ASTs).
   - Extracts path representations between AST leaf tokens.
   - Embeds tokens and AST paths into dense vector spaces.
   - Aggregates path embeddings using attention mechanisms.

2. **Improved LinkNet Architecture (`model.py`)**:
   - Implements **Dilated Convolutions** with formula $ks_{eff} = dr \times (ks - 1) + 1$ (Equation 1).
   - Multi-stage **Encoder-Decoder with Direct Skip Connections** preserving spatial/structural relationships (Equation 2).

3. **Bi-LSTM Sequential Architecture (`model.py`)**:
   - Models forward $\overrightarrow{h_t}$ and backward $\overleftarrow{h_t}$ sequential code context (Equations 3–10).

4. **Score-Level Fusion (`model.py`)**:
   - Fuses LinkNet and Bi-LSTM branch probability scores:
     $$S_{final} = w_1 \cdot S_{linknet} + w_2 \cdot S_{Bi-LSTM}$$
     (Equation 11, where $w_1 = 0.5, w_2 = 0.5$).

5. **Dual Inference Modes (`predict.py` & `app.py`)**:
   - **Mode A: Raw Source Code Snippets** (Python/Java code ➔ AST ➔ Code2Vec ➔ Defect classification).
   - **Mode B: PROMISE Software Metrics** (CM1 McCabe & Halstead metrics).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Test Predictions via CLI
```bash
python predict.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
