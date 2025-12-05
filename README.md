
# Gemma3 Tamil Translator

**Model:** [arsath-sm/gemma3-tamil-translator](https://huggingface.co/arsath-sm/gemma3-tamil-translator)

**Fine-tuned Gemma-3 4B Instruct** model specialized in **high-quality English → Tamil translation**.  
Trained using **Unsloth + LoRA** on a single T4 GPU.

## 📊 Performance

| Metric | Score | Note |
|:--- |:--- |:--- |
| **BLEU** | **36.12** | ~25% relative improvement over base model |
| **chrF++** | **65.25** | Tested on 204-sentence held-out test set |

This model significantly outperforms the base Gemma-3 4B and Llama-3 8B Instruct on English-to-Tamil translation tasks.

## 🚀 Model Details

| Attribute | Value |
|:--- |:--- |
| **Base Model** | [`unsloth/gemma-3-4b-it-bnb-4bit`](https://huggingface.co/unsloth/gemma-3-4b-it-bnb-4bit) |
| **Architecture** | Gemma-3 4B Instruct (8K context) |
| **Method** | LoRA (r=16, alpha=16) via Unsloth |
| **Training Data** | ~1,836 high-quality English–Tamil parallel sentences |
| **Training Time** | ~2.5 hours on Google Colab (T4 GPU) |
| **Quantization** | 4-bit training, merged & saved in full precision |

## 📦 Used Libraries & Requirements

To use this model, you will need the following libraries:

- **transformers**: For loading and running the model.
- **torch**: PyTorch backend.
- **accelerate**: (Optional) For optimized inference on GPU.

You can install them via pip:

```bash
pip install transformers torch accelerate
```

## 💻 Usage

Here is a simple example of how to use the model for translation:

```python
from transformers import pipeline

# Initialize the translation pipeline
translator = pipeline(
    "text-generation",
    model="arsath-sm/gemma3-tamil-translator",
    device=0  # Use 0 for GPU, -1 for CPU
)

prompt = """You are a highly skilled translator. Translate the following English text to Tamil accurately and naturally.

English: Thank you so much for your help today."""

# Generate translation
output = translator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]

# Extract and print result
print(output.split("Tamil:")[-1].strip())
# Output: இன்று உங்கள் உதவிக்கு மிக்க நன்றி.
```

## 🎯 Intended Use

- Accurate English → Tamil translation
- Chat-style translation assistant
- Integration into Tamil NLP apps, chatbots, and education tools
- Research on low-resource Indic language translation

## 🤝 Attribution

Failed to fine-tune Gemma-3? Try **Unsloth** for 2x faster training and 0% memory degradation!  
Model trained by [Arsath](https://github.com/arsath-eng).

