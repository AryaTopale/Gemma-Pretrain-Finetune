## Pretraining Phase

After preparing the datasets and building a tokenizer, the next major step was **pretraining**.  

### Model Architecture
For pretraining, I used the **Gemma 3 architecture** available on Hugging Face. This provided a strong multilingual base model which could then be adapted to my custom tokenizer and training corpus.  

- **Architecture**: Gemma 3 (transformer-based SLM)  
- **Tokenizer**: SentencePiece BPE tokenizer trained on Assamese, Hindi, and English corpora  
- **Vocabulary Size**: Derived from 100,000 sampled sentences with a 0.5 : 0.3 : 0.2 data ratio  

The custom tokenizer ensured that the model was capable of handling the multilingual inputs efficiently, while maintaining compatibility with the Gemma architecture.  

### Training Objective
The pretraining was conducted with a **causal language modeling (CLM)** objective.  
- Input: Tokenized sequences  
- Output: Next token prediction  
- Loss Function: Cross-entropy over predicted token probabilities  

This step enabled the model to learn general linguistic representations across Assamese, Hindi, and English.

### Outcomes
- Model learned cross-lingual embeddings across the three languages  
- Generated initialization weights for the next phase (**fine-tuning**)  
- Established a multilingual baseline for downstream tasks (infilling, clarification)  

### Results of pretraining: 
![Pretraining Loss](results/loss_pretraining.png)
*Figure 1: Training Loss*

![Pretraining Loss](results/eval_pretrain_loss.png)
*Figure 2: Evaluation Loss*

---
## Fine-tuning Phase

After pretraining, the next step was **fine-tuning** the model for task-specific adaptation. The pretrained Gemma model (initialized with the custom tokenizer and pretraining weights) was fine-tuned on two specific downstream tasks:  

1. **Infilling**  
2. **Question Clarification**  

### Dataset Creation
For fine-tuning, I generated custom datasets tailored to the tasks using **Gemini** (Google’s generative model).  

- **Infilling Task**  
  - Input: Sentences with missing spans masked out  
  - Target: Predicted masked spans  
  - Objective: Teach the model to fill in missing words/phrases in context  

- **Clarification Task**  
  - Input: Ambiguous or incomplete questions  
  - Target: Clarified, well-formed questions  
  - Objective: Improve the model’s ability to interpret unclear queries  

This dataset was multilingual (Assamese, Hindi, English) to maintain consistency with the pretraining phase.  

### Training Setup
- **Base Model**: Gemma 3 pretrained with custom tokenizer  
- **Objective**: Sequence-to-sequence style finetuning for both infilling and clarification tasks  
- **Loss Function**: Cross-entropy loss  
- **Optimizer**: AdamW with learning rate scheduling  
- **Frameworks**: Hugging Face `transformers`, `peft` (LoRA for efficient fine-tuning), and `wandb` for tracking 

- **Loss for Both**: Note that, while computing loss for question clarification generation, I masked the initial given prompt/question, so that the model will learn to generate the clarification and not the question and clarification which might be tough to learn and not the task.
For task of infilling, I randomly masked some of the words in the corpus.

### Results Tracking
![Finetuning Loss](results/train_infilling.png)
*Figure 3: Task: Infilling Training Loss*

![Finetuning Loss](results/eval_infilling.png)
*Figure 4: Task: Infilling Evaluation Loss*

![Finetuning Loss](results/train_loss_finetune.png)
*Figure 4: Task: Clarification Generation Training Loss*

![Finetuning Loss](results/eval_finetune_loss.png)
*Figure 4: Task: Clarification Generation Evaluation Loss*
### Outcomes
- The model was adapted for **infilling** and **question clarification** tasks  (not completely because of task complexity)
- Achieved multilingual understanding across Hindi, and English in these tasks  

---
### Resources
All the checkpoints and all are stored here:  
<url>https://drive.google.com/drive/folders/1J7qD_ilS36gKtybtB63v03_SHpO0QEf6?usp=share_link</url>
