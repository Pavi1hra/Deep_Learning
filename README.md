# Deep Learning

A collection of deep learning projects.

---

## Projects

### 1. Text Generation Using LSTM Networks

A word-level language model that generates coherent text continuations from a seed prompt, trained on a literary corpus from Project Gutenberg.

**Key features:**
- Rule-based corpus cleaning (headers, footers, OCR noise removal)
- Three augmentation strategies: synonym replacement, random deletion, and noise injection
- Two-layer LSTM with 200-dimensional GloVe-initialised embeddings and recurrent dropout
- Two-phase training: frozen-embedding warm-up (5 epochs) followed by end-to-end fine-tuning (15 epochs)
- Temperature-scaled sampling for text generation (T = 0.8)

**Results:** The model captures local syntactic patterns (short-range subject-verb and determiner-noun agreement) with degradation in coherence beyond ~15 tokens. Vocabulary coverage limitations result in ~18% [UNKNOWN] token rate in generated output.

**Tech stack:** Python, PyTorch, NLTK

**Usage:**
```python
# Load saved model
model, word2idx = load_model_from_drive(LSTMLanguageModel, load_path)
idx2word = {idx: word for word, idx in word2idx.items()}

# Generate text
prompt = input("Enter a prompt: ")
generated = generate_text(model, prompt, word2idx, idx2word)
print("Generated Text:", generated)
```

**Setup:**
```bash
pip install torch nltk
```

---

### 2. Fine-Grained Bird Species Classification Using Inception-v3

A transfer learning pipeline for 200-class bird species recognition using the CUB-200-2011 benchmark dataset and a pretrained Inception-v3 network fine-tuned in PyTorch.

**Key features:**
- Full fine-tuning of Inception-v3 (ImageNet pretrained) for domain adaptation
- Modified classification head: 2048 → 200 class logits
- Auxiliary classifier retained with weighted loss: `L_total = L_main + 0.4 × L_aux`
- Data augmentation: RandomResizedCrop (299×299), RandomHorizontalFlip, ColorJitter, RandomRotation
- Learning rate scheduling with StepLR / ReduceLROnPlateau and gradient clipping (norm ≤ 5)

**Results:**

| Metric | Score |
|--------|-------|
| Test Accuracy | 81.41% |
| Macro Precision | 81.32% |
| Macro Recall | 81.25% |
| Macro F1-Score | 79.08% |

**Tech stack:** Python, PyTorch, torchvision

**Dataset:** [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) — 11,788 images across 200 bird species.

---

## Repository Structure

```
deep-learning/
├── lstm-text-generation/
│   ├── train.py
│   ├── generate.py
│   └── saved_model/
│       └── lstm_model.pth
├── bird-classification/
│   ├── train.py
│   ├── evaluate.py
│   └── saved_model/
└── README.md
```

## Requirements

```bash
pip install torch torchvision nltk
```

---

## Author

**Pavithra Govinda Raj**  
School of Computing, Data Science and AI, Newcastle University, United Kingdom  
Academic Year 2025–2026
