# ID2223 Lab 2: Fine-tuning Llama 3.2 1B with LoRA

## Overview
Fine-tuned Llama 3.2 1B on FineTome-100k dataset using parameter-efficient 
LoRA technique, achieving 23% loss reduction through systematic optimization.

## Experiments Conducted

### Experiment Summary Table

| Experiment | Steps | LR | Rank | Loss | Improvement | Time (min) |
|-----------|-------|----|----|------|-------------|------------|
| exp1_baseline_60steps | 60 | 2e-4 | 16 | 1.1774 | - | 7.7 |
| exp2_medium_300steps | 300 | 2e-4 | 16 | 0.9736 | ↓17.3% | 13.3 |
| exp3_full_500steps | 500 | 2e-4 | 16 | 0.9414 | ↓20.0% | 22.1 |
| exp4_extended_1000steps | 1000 | 2e-4 | 16 | 0.9068 | ↓23.0% | 43.2 |
| exp5_lr_1e4 | 500 | 1e-4 | 16 | 0.9684 | ↓17.7% | 21.8 |
| exp6_lr_5e4 | 500 | 5e-4 | 16 | 0.9216 | ↓21.7% | 26.7 |
| exp7_lora_r8 | 500 | 2e-4 | 8 | 0.9561 | ↓18.8% | 22.4 |
| exp8_lora_r32 | 500 | 2e-4 | 32 | 0.9286 | ↓21.1% | 22.6 |
| exp9_optimal_combination | 1000 | 5e-4 | 32 | 0.8916 | ↓24.3% | 52.1 |

*exp9 combines best parameters from all experiments*

![alt text](experiments/experiment_comparison.png)

### Key Findings
1. Training duration is the dominant factor (23% improvement)
2. Higher learning rate (5e-4) improves convergence by 2%
3. Larger LoRA rank (r=32) adds 1.4% improvement

[See detailed analysis in EXPERIMENT_REPORT.md](experiments/EXPERIMENT_REPORT.md)

## Optimization Experiments

### Model-Centric Improvements
Conducted systematic experiments on three dimensions:

1. **Training Duration**: Tested 60, 300, 500, 1000 steps
   - Result: Training steps have the most significant impact (23% improvement)
   - Conclusion: Baseline (60 steps) severely undertrained

2. **Learning Rate Tuning**: Tested 1e-4, 2e-4, 5e-4
   - Result: Higher LR (5e-4) achieved 2.1% better loss
   - Conclusion: Model can handle aggressive learning without instability

3. **LoRA Rank Optimization**: Tested r=8, r=16, r=32
   - Result: Larger rank (r=32) improved by 1.4%
   - Conclusion: Better capacity with minimal time overhead

### Data-Centric Analysis
- **Dataset**: FineTome-100k contains 100k instruction-response pairs
- **Observation**: Quality varies; some samples are very short/simple
- **Potential Improvement**: Could filter by quality metrics (perplexity, length)
- **Future Work**: Add domain-specific data for specialized tasks

### Optimal Configuration (exp9)
Combining all best parameters:
- Steps: 1000
- Learning Rate: 5e-4
- LoRA Rank: 32
- Loss: 0.8916

## Deployment
- **Model**: [[HuggingFace Link](https://huggingface.co/kkkkkkatherine/llama-3.2-1b-finetome-1000steps-gguf)]
- **Demo**: [[Spaces Link](https://huggingface.co/spaces/kkkkkkatherine/iris)]
- **Format**: GGUF Q4_K_M (68% size reduction)

## Repository Structure
```
├── experiments/          # All training experiments
│   ├── exp1_baseline_60steps.json
│   ├── exp2_medium_300steps.json
│   ├── exp3_full_500steps.json
│   ├── exp4_extended_1000steps/  # Best model
│   ├── exp5_lr_1e4.json
│   ├── exp6_lr_5e4.json
│   ├── exp7_lora_r8.json
│   ├── exp8_lora_r32.json
│   ├── exp9_optimal_combination.json
│   └── EXPERIMENT_REPORT.md
│   └── experiment_comparison.png
├── notebooks/
│   └── lab2.ipynb
│   └── lab2_experiment.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Conclusions
Best configuration: 1000 steps, lr=5e-4, r=32
Achieves 23% loss improvement over baseline while maintaining 
efficient training time (~45 min).
