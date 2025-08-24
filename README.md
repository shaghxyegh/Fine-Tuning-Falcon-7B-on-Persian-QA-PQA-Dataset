# Fine-Tuning-Falcon-7B-on-Persian-QA-PQA-Dataset
My first fine-tuning attempt on Falcon-7B for Persian QA using Colab T4 GPU. Overcame dataset loading errors (RuntimeError on scripts), overfitting (adjusted epochs), and T4 VRAM limits. Training didn't complete due to model size. Achieved partial run (step 51/125). Challenges: Transformers complexities. Future: Complete on A100, evaluate F1/EM
# Project Report: Fine-Tuning Falcon-7B on Persian QA (PQA) Dataset

## Project Overview
This was my first attempt to fine-tune a model, using Google Colab’s free T4 GPU to adapt Falcon-7B-Instruct with 4-bit quantization and LoRA for Persian question-answering on the PQA dataset. The report is in English, focusing on Persian text processing.

**Objectives**:
- Load and preprocess PQA dataset for Persian QA.
- Fine-tune Falcon-7B using LoRA and quantization.
- Address Persian text challenges (e.g., RTL).
- Prepare for evaluation (incomplete due to resource limits).

**Tech Stack**:
- Python libraries: Transformers, Datasets, PEFT, BitsAndBytes, TRL.
- Model: tiiuae/falcon-7b-instruct (4-bit quantized).
- Environment: Google Colab (free T4 GPU, ~15GB VRAM).

## Methodology
1. **Data Preparation**:
   - Loaded PQA train (901 examples) and test (93 examples) JSON files.
   - Flattened into SQuAD format (`id`, `context`, `question`, `answers`).
   - Tokenized with Falcon tokenizer, formatted for causal LM, subsampled (1000 train, 200 eval).

2. **Model Setup**:
   - Loaded Falcon-7B with 4-bit quantization (BitsAndBytes) for T4 compatibility.
   - Enabled gradient checkpointing to save memory.

3. **Fine-Tuning**:
   - Configured LoRA (r=16, target_modules=["q_proj", "v_proj"]).
   - Training args: 1 epoch, batch size 1, gradient accumulation 8, BF16, eval every 50 steps.
   - Used SFTTrainer for supervised fine-tuning.

4. **Training Execution**:
   - Ran training, but it stopped at step 51/125 (Epoch 0.40/1) due to model size and T4 limits.

## Results
- **Training Progress**: Reached step 51/125 (Epoch 0.40/1) in ~2.5 hours, with decreasing training loss.
- **Incomplete Training**: Due to the huge 7B model size, training didn’t complete on Colab T4 (VRAM limits, timeouts).
- **No Evaluation**: Couldn’t compute metrics (e.g., Exact Match, F1) due to incomplete training.

## Challenges and Solutions
1. **Dataset Loading Error**:
   - **Challenge**: Direct dataset loading failed with `RuntimeError: Dataset scripts are no longer supported, but found pquad.py`. JSON access also failed.
   - **Solution**: Manually downloaded and uploaded JSON files to Colab for direct loading.

2. **Overfitting Risk**:
   - **Challenge**: Small PQA dataset (~1000 examples) risked overfitting; model showed early signs of memorization.
   - **Solution**: Limited to 1 epoch to balance learning and generalization. Monitored loss trends.

3. **Colab T4 Limitations**:
   - **Challenge**: T4’s limited VRAM (~15GB) and timeouts couldn’t handle the 7B model, stopping training early.
   - **Solution**: Used 4-bit quantization and gradient accumulation (8 steps). Partial run achieved.

4. **Transformers Library Complexity**:
   - **Challenge**: Latest Transformers version had complexities (e.g., deprecated checkpointing, renamed `eval_strategy`); compatibility issues with other libraries.
   - **Solution**: Adapted to `eval_strategy`, ignored warnings, and used compatible versions (e.g., `peft`, `trl`).

## Future Work
- **Complete Training**: Use a high-VRAM GPU (e.g., A100 via Colab Pro, AWS, or GCP) to finish training. Increase gradient accumulation (e.g., 16 steps) or use DeepSpeed for distributed training. Resume from saved checkpoints (`./falcon7b-pqa-lora`).
- **Evaluation**: Post-training, compute SQuAD metrics (Exact Match, F1) on the full test set.
- **Persian Optimization**: Fine-tune with Persian-specific tokenizers or augment data with translations.
- **Model Merging**: Merge LoRA adapters with base model for deployment.
- **Deployment**: Create a Hugging Face or FastAPI endpoint for Persian QA.

## References
- PQA Dataset: https://huggingface.co/datasets/SajjadAyoubi/persian_qa.
- Hugging Face Docs: https://huggingface.co/docs/transformers
- Model: https://huggingface.co/tiiuae/falcon-7b-instruct
- PEFT Docs: https://huggingface.co/docs/peft
- TRL Docs: https://huggingface.co/docs/trl

**Author**: Shaghayegh Shafiee  
**Date**: August 24, 2025
