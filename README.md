# 💰 FinanceExplain: A Domain-Specific Financial Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UoonpR4aabQe693J_3gGjMlOJbowC3ad#scrollTo=j3GheOw6mQu1)

## 📌 Project Overview

**FinanceExplain** is a domain-specific conversational assistant built by fine-tuning a pretrained large language model (LLM) for finance tasks such as:
- 🔎 Explaining complex financial sentences in simple terms.
- 📘 Defining financial terms in plain language.

The model is adapted using **LoRA (Low-Rank Adaptation)** — a parameter-efficient fine-tuning (PEFT) technique — enabling efficient training on limited compute resources (e.g., Google Colab GPUs).  
This project demonstrates dataset creation & preprocessing, fine-tuning experimentation, evaluation using multiple metrics, and deployment via a user-friendly Gradio interface.

---

## 🎯 Project Goals

- Build a **finance-focused LLM** that understands and responds to domain queries.
- Use **parameter-efficient fine-tuning** to adapt the model with limited resources.
- Evaluate performance using **BLEU, ROUGE-L, F1, perplexity, and qualitative tests**.
- Deploy an intuitive **UI for real users** to interact with the assistant.

---

## 🧠 Model & Tools

- **Base Model:** Pretrained causal LM (e.g., TinyLlama/Gemma style) from Hugging Face.
- **Fine-Tuning Method:** LoRA via the `peft` library.
- **Libraries Used:**  
  `transformers`, `datasets`, `peft`, `bitsandbytes`, `evaluate`, `torch`, `gradio`.
- **Hardware:** Google Colab with GPU support.

---

## 📚 Dataset

1. **Financial PhraseBank** (Malo et al., 2014):  
   A widely used financial text dataset with sentiment labels.
2. **Custom Definition Set:**  
   Curated definitions of common financial terms (e.g., profit margin, revenue, liquidity, assets).

**Instruction-Response Formatting**  
All examples were converted into clearly structured templates such as:
<s>[INST] Explain this financial sentence in simple terms:
{sentence} [/INST] {model_response}</s>

and

<s>[INST] Define the financial term in simple words:
{term} [/INST] {definition}</s>


---

## 🧹 Data Preprocessing

1. Removed duplicate and invalid samples.
2. Normalized text (stripping, lower-casing in prep steps).
3. Tokenized using model’s own tokenizer (subword tokenization).
4. Ensured context windows fit within model limits.
5. Shuffled and split into training & validation sets.

These preprocessing steps ensure the dataset is clean, consistent, and ready for domain adaptation.

---

## 🏋️ Fine-Tuning Strategy

**Parameter-Efficient LoRA Setup:**
- Low-rank adapters inserted into attention layers (`q_proj`, `v_proj`).
- Rank `r = 8`, `alpha = 16`, dropout = 0.05.
- Only LoRA parameters trained; base model frozen.

**Hyperparameter Variants Tested:**
- Learning rates: `1e-4`, `5e-5`, `2e-5`
- Epoch counts: `1`, `2`, `3`
- Batch sizes small due to GPU limits.

---

## 🧪 Experiments & Results

| Experiment | LR   | Epochs | Val Loss | Perplexity | ROUGE-L | BLEU  | F1   |
|------------|------|--------|----------|------------|---------|-------|------|
| Exp1       | 1e-4 | 1      | 0.396    | 1.49       | 0.780   | 69.22 | ~0.65 |
| Exp2       | 5e-5 | 2      | 0.378    | 1.46       | 0.784   | 69.76 | ~0.66 |
| Exp3       | 2e-5 | 2      | 0.648    | 1.91       | 0.663   | 49.34 | ~0.52 |
| **Exp4**   | **5e-5** | **3** | **0.351** | **1.42** | **0.793** | **70.80** | **0.676** |

✨ **Best configuration:** Exp4  
→ Lowest loss, lowest perplexity, highest BLEU/ROUGE/F1.

---

## 📊 Evaluation Metrics

- **Validation Loss:** How well the model fits held-out data.
- **Perplexity:** Model’s confidence at predicting text (`exp(loss)`, lower is better).
- **BLEU:** N-gram overlap metric.
- **ROUGE-L:** Longest common subsequence overlaps.
- **F1-score:** Token precision/recall balance between outputs & references.
- **Qualitative Tests:** Manual comparisons showing fine-tuned model is far more accurate than base.

**Key Observation:**  
The **base model** often repeats/mimics instruction text without following the task, whereas the **fine-tuned model** produces clear, domain-specific outputs.

---

## 🖥️ Interactive UI (Gradio)

The interface:
- Automatically detects whether input is a definition or sentence explanation.
- Validates that input contains financial content.
- Displays formatted, human-readable outputs.
- Rejects non-finance queries with helpful prompts.


---

## ▶️ How to Run (Google Colab)

1. Open the badge link above.
2. Select **GPU** under Runtime → Change runtime type.
3. Run cells in order:
   - Install dependencies
   - Load & preprocess dataset
   - Train LoRA adapters
   - Evaluate
   - Launch UI
4. Interact with the Gradio interface.


---

## 📚 References (APA)

- Abid, A., Abdalla, A., Abid, A., et al. (2019). *Gradio: Hassle-free sharing and testing of ML models in the browser.*  
- Hu, E. J., Shen, Y., Wallis, P., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR).  
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing (3rd ed. draft).*  
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.  
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.* Journal of the Association for Information Science and Technology.  
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation.* ACL.  
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* NeurIPS.  
- Wolf, T., Debut, L., Sanh, V., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP: System Demonstrations.

---

## 🧾 Summary

FinanceExplain is an example of:
✔ Successful domain adaptation using LoRA  
✔ Efficient training on limited hardware  
✔ Clear improvements over baseline models  
✔ Multi-metric evaluation  
✔ User-friendly deployment

Feel free to explore, adapt, and extend this system for other specialized domains!
