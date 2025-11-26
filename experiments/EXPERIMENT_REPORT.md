# Lab 2 Experiment Results Report
Generated: 2025-11-25 14:41:13

## Overview
Total experiments conducted: 8

## Experiment Comparison Table

### Training Steps Comparison
| Experiment | Steps | Learning Rate | LoRA Rank | Final Loss | Training Time (min) |
|-----------|-------|--------------|-----------|------------|---------------------|
| exp1_baseline_60steps | 60 | 2e-04 | 16 | 1.1774 | 7.71 |
| exp2_medium_300steps | 300 | 2e-04 | 16 | 0.9736 | 13.26 |
| exp3_full_500steps | 500 | 2e-04 | 16 | 0.9414 | 22.13 |
| exp4_extended_1000steps | 1000 | 2e-04 | 16 | 0.9068 | 43.20 |
| exp5_lr_1e4 | 500 | 1e-04 | 16 | 0.9684 | 21.79 |
| exp6_lr_5e4 | 500 | 5e-04 | 16 | 0.9216 | 26.66 |
| exp7_lora_r8 | 500 | 2e-04 | 8 | 0.9561 | 22.38 |
| exp8_lora_r32 | 500 | 2e-04 | 32 | 0.9286 | 22.61 |

## Detailed Results

### exp1_baseline_60steps

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 60
- learning_rate: 0.0002
- lora_r: 16
- lora_alpha: 16
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 1.1774
- total_steps: 60
- training_time_seconds: 462.3720
- training_time_minutes: 7.7062

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve their performance over time, without being explicitly programmed for each task. It involves traini...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning involves training a model on a labeled dataset, where the correct output is already known. The goal is to learn a mapping between input data and output labels.

Unsupervised learni...

---

### exp2_medium_300steps

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 300
- learning_rate: 0.0002
- lora_r: 16
- lora_alpha: 16
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9736
- total_steps: 300
- training_time_seconds: 795.8442
- training_time_minutes: 13.2641

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve their performance over time, without being explicitly programmed for each task. It involves traini...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning and unsupervised learning are two types of machine learning algorithms. In supervised learning, the algorithm is trained to predict a specific target variable based on a set of lab...

---

### exp3_full_500steps

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 500
- learning_rate: 0.0002
- lora_r: 16
- lora_alpha: 16
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9414
- total_steps: 500
- training_time_seconds: 1328.0504
- training_time_minutes: 22.1342

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve their performance over time, without being explicitly programmed for each task. It involves traini...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning is a type of machine learning where the algorithm is trained to recognize and classify patterns in the data. The algorithm is trained on labeled data, where each instance has a cor...

---

### exp4_extended_1000steps

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 1000
- learning_rate: 0.0002
- lora_r: 16
- lora_alpha: 16
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9068
- total_steps: 1000
- training_time_seconds: 2591.7747
- training_time_minutes: 43.1962

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to automatically learn from data. It involves training algorithms on large datasets, so they can make predictions or de...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning is a type of machine learning where the algorithm is trained to recognize and classify patterns in the data. The algorithm is given labeled data, and the goal is to learn from this...

---

### exp5_lr_1e4

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 500
- learning_rate: 0.0001
- lora_r: 16
- lora_alpha: 16
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9684
- total_steps: 500
- training_time_seconds: 1307.1076
- training_time_minutes: 21.7851

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve their performance over time, without being explicitly programmed for each task. It involves traini...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning and unsupervised learning are two types of machine learning algorithms. In supervised learning, the algorithm is trained on a set of labeled data, where each instance has a corresp...

---

### exp6_lr_5e4

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 500
- learning_rate: 0.0005
- lora_r: 16
- lora_alpha: 16
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9216
- total_steps: 500
- training_time_seconds: 1599.6847
- training_time_minutes: 26.6614

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to automatically learn from data, without being explicitly programmed. It uses algorithms to analyze data, identify pat...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning involves a labeled dataset, where each sample is classified into a specific class. The goal is to predict the class of each sample. In contrast, unsupervised learning does not requ...

---

### exp7_lora_r8

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 500
- learning_rate: 0.0002
- lora_r: 8
- lora_alpha: 8
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9561
- total_steps: 500
- training_time_seconds: 1342.9305
- training_time_minutes: 22.3822

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve their performance over time, without being explicitly programmed for each task. It involves traini...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning is a type of machine learning where the algorithm is trained to recognize and classify patterns in the data. The algorithm is trained on labeled data, where each instance has a cor...

---

### exp8_lora_r32

**Configuration:**
- model_name: Llama-3.2-1B-Instruct
- max_steps: 500
- learning_rate: 0.0002
- lora_r: 32
- lora_alpha: 32
- batch_size: 2
- gradient_accumulation_steps: 4

**Metrics:**
- final_loss: 0.9286
- total_steps: 500
- training_time_seconds: 1356.5755
- training_time_minutes: 22.6096

**Sample Outputs:**

**Q1:** What is machine learning?

**A1:** Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve their performance over time, without being explicitly programmed for each task. It involves traini...

**Q2:** Explain the difference between supervised and unsupervised learning.

**A2:** Supervised learning is a type of machine learning where the algorithm is trained to recognize and classify patterns in the data. The algorithm is given labeled data, and the goal is to learn a mapping...

---

## Conclusions

### Key Findings:

#### 1. Training Steps Impact (MOST SIGNIFICANT)
- **60 steps**: Loss = 1.1774 (severely undertrained)
- **300 steps**: Loss = 0.9736 (17.3% improvement)
- **500 steps**: Loss = 0.9414 (20.0% improvement)
- **1000 steps**: Loss = 0.9068 (23.0% improvement)

**Conclusion**: Training steps have the most dramatic impact on model quality. 
The baseline (60 steps) was clearly insufficient. Loss continues to improve 
even at 1000 steps, suggesting more training could help further.

#### 2. Learning Rate Analysis
Testing at 500 steps with different learning rates:
- **1e-4 (low)**: Loss = 0.9684 (slower convergence, cautious)
- **2e-4 (baseline)**: Loss = 0.9414 (balanced)
- **5e-4 (high)**: Loss = 0.9216 (fastest convergence)

**Conclusion**: Higher learning rate (5e-4) achieved 2.1% better loss than 
baseline 2e-4, with similar training time. The model can handle more 
aggressive learning without instability.

#### 3. LoRA Rank Impact
Testing at 500 steps with different ranks:
- **r=8**: Loss = 0.9561 (limited capacity)
- **r=16**: Loss = 0.9414 (good balance)
- **r=32**: Loss = 0.9286 (best performance, only 2% more time)

**Conclusion**: Larger LoRA rank provides better model capacity with minimal 
time overhead. r=32 achieved 1.4% improvement over r=16.

### Recommendations:

Based on experimental evidence:

1. **Always train for 500+ steps minimum** (60-300 steps insufficient)
2. **Use higher learning rate (5e-4)** for faster convergence
3. **Use larger LoRA rank (r=32)** if training time permits
4. **For quick iterations**: 500 steps with r=16 is acceptable
5. **For production**: 1000 steps with r=32 and lr=5e-4

### Best Configuration:

**Optimal setup for quality:**
- **Steps**: 1000 (or more if time allows)
- **Learning Rate**: 5e-4 (faster than baseline 2e-4)
- **LoRA Rank**: 32 (better capacity than 16)
- **Estimated Performance**: Loss ~0.88-0.90 (extrapolating)
- **Training Time**: ~45-50 minutes

**Reasoning**: 
1. Training steps show diminishing returns but still improve at 1000
2. Higher learning rate (5e-4) converges faster without instability
3. Larger rank (r=32) captures more model capacity with minimal overhead
4. This combination should achieve best quality/time tradeoff

### Deployment Model:
Selected **exp4_extended_1000steps** (Loss: 0.9068) for deployment as it 
represents the best-trained model in our experiments.
