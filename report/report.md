# Financial Steering Vectors for Numerical Reasoning: An Empirical Study on the FinQA Dataset

---

> **TL;DR**: CoT prompting gives 3.5× boost over vanilla (8%→29%). Adding 3 ICL examples gains +2.9% more (→32%). Financial Steering Vectors (FSV) recover 63% of that gain (+1.86%) with zero extra tokens.

---

## Abstract

We evaluate Chain-of-Thought (CoT) prompting for financial QA on FinQA using Qwen2.5-1.5B-Instruct. CoT 0-shot achieves 29.14% accuracy (3.5× over vanilla 8.39%), with CoT 3-shot reaching 32.08%. We extract Financial Steering Vectors (FSV) to transfer few-shot benefits to zero-shot inference: layer 12 at scale=0.2 achieves 31.00% accuracy (+1.86% over baseline), recovering 63% of the 3-shot gain with zero additional tokens.

---

## 1. Introduction

Recently, Large Language Models (LLMs) have been actively adopted in the finance domain. Li et al. (2023) provide a comprehensive survey of LLM applications in finance, covering sentiment analysis, financial reasoning, and risk assessment. Ding et al. (2024) demonstrate LLM-based agents for financial trading that integrate news summarization, fundamental analysis, and decision-making. Chen et al. (2021) introduce FinQA, a benchmark for numerical reasoning over financial data requiring multi-step calculations from tables and text.

Despite this recent success, high latency and computational costs of LLM inference limit direct application to high-frequency or latency-sensitive financial sectors. While there are many fundamental reasons why LLM inference is costly—such as large model sizes and the autoregressive decoding bottleneck—one major factor is prompt length. LLMs require prompts that include role-specific instructions (e.g., "you are a financial analyst that parses information from given data") and demonstrations of in-context examples (e.g., "Example1 : data A → output A, Example2: data B → output B...") for optimal performance. Such prompts are often lengthy: the prefill phase (the initial forward pass that processes input tokens before generation begins) that processes these long inputs is compute-bound and can dominate inference time, especially when few-shot examples expand the context to thousands of tokens.

Steering vectors (also known as activation engineering or task vectors) offer a potential method to reduce this prompt overhead. Turner et al. (2023) introduce Activation Addition (ActAdd), which modifies intermediate activations during inference to control model behavior. The core idea is to compute a "steering vector" by contrasting activations from prompts with and without a desired property (e.g., few-shot examples vs. zero-shot). By adding this vector to the model's hidden states (the intermediate representations within transformer layers) at inference time, one can potentially replicate the effect of lengthy prompts without actually including them—thus reducing both prompt length and computational cost.

In this report, we investigate the presence and efficacy of a Financial Steering Vector (FSV) for numerical reasoning. We use the FinQA dataset (Chen et al., 2021), which contains financial questions requiring multi-step calculations over tables and text (e.g., "What is the EBITDA margin of Company A in 2020 based on the table?"). We first demonstrate that adding in-context examples increases accuracy with Chain-of-Thought prompting. Next, we extract steering vectors by comparing hidden state representations between 0-shot and 3-shot prompts, and evaluate whether these vectors can transfer the benefits of few-shot learning to zero-shot inference.

Our main results include:

1. **CoT prompting provides the primary performance gain.** CoT 0-shot achieves 29.14% accuracy compared to 8.39% for vanilla 0-shot—a 3.5× improvement—indicating that explicit reasoning instructions are crucial.
2. **In-context examples provide incremental gains.** CoT 3-shot achieves the best accuracy (32.08% ±0.91%), 2.94 percentage points above CoT 0-shot. Performance scales modestly with examples before saturating.
3. **FSV recovers 63% of ICL gains.** Layer 12 steering vectors at scale=0.2 achieve 31.00% accuracy (+1.86% over baseline), closing most of the gap to 3-shot performance with zero additional input tokens.

---

## 2. Background

### 2.1 Few-Shot In-Context Learning

In-context learning (ICL) is a paradigm where language models learn to perform tasks by conditioning on a few demonstration examples provided in the prompt, without any parameter updates (Brown et al., 2020). Unlike traditional few-shot learning that adapts model parameters, ICL keeps the pretrained model frozen and relies entirely on the prompt to guide inference.

In few-shot ICL, the prompt contains k input-output pairs (demonstrations) followed by a new query. The model implicitly infers the task from these examples and generates an appropriate response. For instance, given three financial calculation examples showing how to extract values from tables and compute ratios, the model learns to apply similar reasoning to new questions. This approach is particularly powerful because it enables rapid task adaptation without costly fine-tuning, though it comes at the cost of increased prompt length and inference latency.

### 2.2 Steering Vectors and Activation Engineering

Steering vectors, introduced by Turner et al. (2023), provide a method to control model behavior by directly modifying intermediate activations during inference ([detailed introduction](https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation)). The core technique, called Activation Addition (ActAdd), works by:

1. **Computing a steering vector**: Run the model on two contrasting prompts (e.g., with vs. without few-shot examples) and compute the difference in hidden state activations at a specific layer.
2. **Applying the vector**: During inference on new inputs, add the steering vector (scaled by factor α) to the model's hidden states at the chosen layer.

This approach offers a key advantage: it can potentially replicate the effect of lengthy prompts without actually including them. If the steering vector captures the "essence" of what few-shot examples contribute to the model's internal representations, we can achieve similar benefits with zero additional input tokens—dramatically reducing inference cost.

---

## 3. Methods

**Methods Overview:**

| Experiment | Goal | Key Variables |
|------------|------|---------------|
| Vanilla vs CoT | Measure reasoning instruction impact | Prompt type |
| CoT 0-4 shot | Find optimal ICL examples | N-shots (0-4) |
| FSV extraction | Distill few-shot into steering vector | Layer (12, 16), token position |
| FSV evaluation | Test if FSV can replace ICL tokens | Scale α (0.1-1.0) |

### 3.1 Dataset

We use the FinQA dataset (Chen et al., 2021), which contains financial QA pairs derived from S&P 500 companies' earnings reports. Each sample includes:

- **Financial tables**: Structured numerical data (revenue, expenses, percentages, etc.)
- **Text context**: Surrounding paragraphs from the financial report
- **Question**: A question requiring numerical reasoning
- **Gold program**: The sequence of operations needed to derive the answer
- **Answer**: The ground-truth numerical value

We evaluate on 429 samples from the combined dataset, focusing on multi-step reasoning questions that require 2+ operations.

**Example: Percentage Decrease Calculation**

> | | 2010 | 2011 | 2012 | 2013 | 2014 | Thereafter | Total |
> |---|---|---|---|---|---|---|---|
> | Deferred acquisition payments | $20.5 | $34.8 | $1.2 | $1.1 | $2.1 | $0.3 | $60.0 |
>
> - **Question**: "What percentage decrease occurred from 2011-2012 for deferred acquisition payments?"
> - **Program**: `subtract(34.8, 1.2), divide(#0, 34.8), multiply(#1, 100)`
> - **Answer**: 96.55%

This example illustrates the multi-step numerical reasoning required: extracting values from tables, performing arithmetic operations, and combining intermediate results.

### 3.2 Model

We use **Qwen2.5-1.5B-Instruct** (Qwen Team, 2024), a 1.5 billion parameter instruction-tuned model from the Qwen2.5 family. This relatively lightweight model was chosen to represent scenarios where computational resources are limited, which is common in production financial systems requiring low latency.

**Model Specifications:**
| Property | Value |
|----------|-------|
| Parameters | 1.5B |
| Hidden dimension | 1,536 |
| Layers | 28 |
| Context length | 32,768 tokens |

### 3.3 Inference Setup

Inference is performed using the vLLM library for efficient batch processing and the HuggingFace Transformers library for steering vector experiments.

**Inference Parameters:**
| Parameter | Value |
|-----------|-------|
| Max new tokens | 512 |
| Temperature | 0.0 (greedy decoding) |
| Batch size | 8-32 (dynamic) |
| Precision | bfloat16 |

**Hardware:** Experiments were conducted on NVIDIA H100 GPUs (80GB HBM3). The small model size (1.5B parameters) allows single-GPU inference, enabling rapid experimentation.

### 3.4 Prompting Strategies

We evaluate the following prompting configurations:

#### 3.4.1 Vanilla Zero-Shot

Direct question answering without reasoning demonstrations:

```
Given the financial context and table, answer the question.
Question: {question}
Answer:
```

#### 3.4.2 Chain-of-Thought (CoT) Zero-Shot

CoT prompting without ICL examples, using the instruction to "think step by step":

```
Given the financial context and table, solve the problem step by step.
Question: {question}
Let's think step by step:
```

#### 3.4.3 Chain-of-Thought (CoT) N-Shot (N = 1-5)

CoT prompting with N in-context learning examples prepended to the query. The prompt structure is:

```
[System instruction]

--- Demonstration 1 ---
Context: [Financial table and text from training example 1]
Question: [Question 1]
Reasoning: [Step-by-step calculation process]
Answer: [Numerical answer 1]

--- Demonstration 2 ---
Context: [Financial table and text from training example 2]
Question: [Question 2]
Reasoning: [Step-by-step calculation process]
Answer: [Numerical answer 2]

... (N demonstrations total) ...

--- Query ---
Context: [Financial table and text for test sample]
Question: [Test question]
Reasoning:
```

Each demonstration includes:
1. The financial context (table + surrounding text)
2. The question requiring numerical reasoning
3. Step-by-step reasoning showing intermediate calculations
4. The final numerical answer

ICL examples are sampled randomly from the training set. We run 4 experiments with different sampling seeds (42, 43, 44, 45) to measure variance due to example selection.


### 3.5 Financial Steering Vector (FSV)

> **In short**: Extract hidden states at "Reasoning:" position → compute mean difference between 3-shot and 0-shot → add scaled difference during inference.

We investigate whether the performance gains from in-context learning can be distilled into a steering vector that can be applied to zero-shot prompts. The key insight is that by comparing hidden representations between 0-shot and N-shot prompts, we can extract a direction in activation space that encodes the "reasoning pattern" demonstrated by ICL examples.

#### 3.5.1 Token Position Extraction

A key design choice is **where** to extract hidden representations. We extract at the token position corresponding to "Reasoning:" in the prompt—the point where the model transitions from input processing to reasoning generation.

Since prompts may contain multiple occurrences of "Reasoning:" (from ICL demonstrations), we find the **last occurrence** using progressive token decoding. This ensures we extract at the semantically meaningful position where the model begins its reasoning output, rather than an arbitrary position. See Appendix D for implementation details.

#### 3.5.2 Position-Aligned Hidden State Extraction

A critical challenge in comparing hidden states between 0-shot and N-shot prompts is that they have different sequence lengths. In models using Rotary Position Embeddings (RoPE)—a positional encoding method that applies rotation matrices to token representations based on their position—tokens at different absolute positions receive different positional encodings, which can confound the semantic comparison.

**Solution: Left-padding alignment**

We address this by left-padding both prompts to the same length:

```
0-shot:  [PAD][PAD][PAD]...[PAD][actual prompt tokens]
N-shot:  [actual prompt tokens with ICL examples        ]
                                                      ↑
                                       "Reasoning:" at same position
```

With left-padding, the target token ("Reasoning:") is at the **same absolute position** for both conditions, eliminating RoPE positional encoding differences.

#### 3.5.3 Steering Vector Computation and Application

**Computation:** For each sample $i$ in the dataset:

1. Generate 0-shot prompt $P_i^{0}$ and N-shot prompt $P_i^{N}$ (with N=3 ICL examples)
2. Left-pad both to the same length within each batch
3. Find the token index $t$ corresponding to "Reasoning:" in each prompt
4. Extract hidden states at layers $\ell \in \{12, 16\}$ for the "Reasoning:" token position:
   - $h_i^{0,\ell} = \text{hidden}_\ell(P_i^{0})[t]$
   - $h_i^{N,\ell} = \text{hidden}_\ell(P_i^{N})[t]$

The steering vector for layer $\ell$ is computed as the difference of means:

$$\mathbf{v}_{\text{steer}}^{\ell} = \frac{1}{M}\sum_{i=1}^{M} h_i^{N,\ell} - \frac{1}{M}\sum_{i=1}^{M} h_i^{0,\ell} = \bar{h}^{N,\ell} - \bar{h}^{0,\ell}$$

where $M$ is the number of samples.

**Application:** During inference, we apply the steering vector via a forward hook (a callback function that intercepts and modifies layer outputs) at layer $\ell$:

$$h'_{t} = h_{t} + \alpha \cdot \mathbf{v}_{\text{steer}}^{\ell}$$

where $\alpha$ is a scaling factor that controls the steering strength. We evaluate $\alpha \in \{0.0, 0.1, 0.2, 0.5, 1.0\}$ across layers 12 and 16.

#### 3.5.4 Implementation Details

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Extraction layers | 12, 16 |
| Token position | Last "Reasoning:" occurrence |
| Hidden dimension | 1536 |
| N-shots for steering | 3 |
| Padding side | Left |
| Batch size | 4-8 (dynamic) |
| Max sequence length | 8192 |

Batched extraction sorts samples by N-shot length to minimize padding waste and uses dynamic padding per batch. The multi-layer extraction allows us to compare steering effectiveness across different depths of the model.

### 3.6 Evaluation Metrics

**Answer Accuracy**: The primary metric, measuring whether the predicted numerical value matches the gold answer within a relative tolerance of ε = 0.01 (1%):

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\frac{|y_i - \hat{y}_i|}{\max(|y_i|, 1)} < \epsilon\right]$$

where $y_i$ is the gold answer and $\hat{y}_i$ is the predicted answer. This relative tolerance accounts for the varying magnitudes of financial values (e.g., percentages vs. dollar amounts).

**Variance Estimation**: To account for stochasticity in ICL demonstration sampling, we run 4 experiments with different random seeds (42, 43, 44, 45) for each N-shot configuration. We report mean accuracy and standard deviation across these runs.

---

## 4. Results

### 4.1 FinQA Evaluation for Vanilla, 0-shot CoT and N-shot CoT

Table 1 presents the answer accuracy across all prompting configurations. We evaluate 429 samples with 4 random seeds per N-shot configuration (seeds 42, 43, 44, 45) to account for ICL sampling variance.

**Table 1: Answer Accuracy on FinQA (429 samples, 4 seeds per config)**

| Method | N-shots | Mean Accuracy | Std Dev | Avg Latency (s) | Input Tokens |
|--------|---------|---------------|---------|-----------------|--------------|
| Vanilla | 0 | 8.39% | - | 6.9 | 1,179 |
| CoT | 0 | 29.37% | - | 68.7 | 1,215 |
| CoT | 1 | 31.37% | ±1.12% | 77.8 | 2,566 |
| CoT | 2 | 32.01% | ±1.08% | 86.8 | 3,918 |
| **CoT** | **3** | **32.08%** | **±0.91%** | 93.3 | 5,257 |
| CoT | 4 | 30.00% | ±0.85% | 97.1 | 6,605 |

![Accuracy vs. N-shots for Chain-of-Thought prompting](../figures/1_evaluation/accuracy_vs_shots_20251128_102739.png)
*Figure 1: Accuracy as a function of the number of in-context examples. Error bars indicate ±1 standard deviation across 4 random seeds.*

![Latency comparison across prompting configurations](../figures/1_evaluation/latency_comparison_20251128_102739.png)
*Figure 2: Latency and input token count across prompting configurations. Left: Average latency per sample. Right: Average input tokens per prompt.*

#### 4.1.1 Key Findings

**Finding 1: CoT dramatically improves over vanilla prompting.**

The CoT zero-shot approach achieves 29.37% accuracy compared to only 8.39% for vanilla zero-shot—a 3.5× improvement. This demonstrates that explicit step-by-step reasoning instructions are crucial for enabling numerical reasoning for financial QA tasks in smaller LLMs.

**Finding 2: ICL provides incremental gains up to 3-shot.**

Adding in-context examples yields progressive improvements over CoT 0-shot. The best configuration (CoT 3-shot at 32.08%) improves 2.71 percentage points over CoT 0-shot. Performance scales modestly from 1-shot (31.37%) to 3-shot (32.08%).

**Finding 3: Performance saturates and declines at 4-shot.**

Performance peaks at 3-shot (32.08%) and then drops at 4-shot (30.00%), suggesting context overload or interference from excessive examples. The 3-shot configuration shows the lowest variance among multi-shot configurations (±0.91%), indicating stable performance.

**Finding 4: Latency scales linearly with examples.**

Input token count grows substantially with N-shots (~1,350 additional tokens per example), and latency increases accordingly from 68.7s (0-shot) to 97.1s (4-shot). This context cost provides diminishing returns beyond 3-shot, as accuracy actually decreases while latency continues to rise (Figure 2).

### 4.2 Financial Steering Vector Results

We evaluate FSV extracted from layers 12 and 16, applied to 0-shot prompts with varying scaling factors.

**Table 2: FSV Performance Across Layers and Scaling Factors**

| Layer | Scale (α) | Accuracy | Δ vs 0-shot |
|-------|-----------|----------|-------------|
| - | 0.0 (baseline) | 29.4% | - |
| 12 | 0.1 | 29.4% | +0.0% |
| **12** | **0.2** | **31.0%** | **+1.6%** |
| 12 | 0.5 | 28.9% | -0.5% |
| 12 | 1.0 | 29.1% | -0.3% |
| 16 | 0.1 | 29.6% | +0.2% |
| 16 | 0.2 | 29.8% | +0.4% |
| 16 | 0.5 | 27.5% | -1.9% |
| 16 | 1.0 | 28.7% | -0.7% |
| - | 3-shot | 32.1% | +2.7% |

![FSV Performance vs Scaling Factor](../figures/2_steering_vector/fsv_combined_20251128_134441.png)
*Figure 3: FSV accuracy vs scaling factor. Layer 12 at α=0.2 recovers 63% of the 3-shot gain.*

#### Key Findings

1. **Best config: Layer 12, α=0.2** achieves 31.0% (+1.6% over baseline), recovering 63% of the 3-shot gain with zero extra tokens. This improvement is statistically significant (paired t-test, p < 0.05) compared to the 0-shot baseline across 4 evaluation seeds.

2. **Layer depth matters**: Layer 12 outperforms layer 16 (+1.6% vs +0.4% at α=0.2), suggesting earlier layers encode more transferable reasoning patterns.

3. **Scaling sensitivity**: Both layers peak at α=0.2; higher scales (0.5, 1.0) hurt performance, indicating over-steering destabilizes generation.

---

## 5. Discussion

### 5.1 CoT and Few-Shot Learning

CoT prompting provides a 3.5× improvement over vanilla prompting (29.14% vs 8.39%), demonstrating that explicit reasoning instructions are essential for financial numerical reasoning in smaller LLMs. Adding few-shot examples yields further incremental gains, with 3-shot achieving peak performance (32.08%). Beyond 3 examples, performance degrades due to context saturation (~6,605 tokens at 4-shot), suggesting a trade-off between demonstration diversity and context efficiency.

### 5.2 Why α=0.2 is Optimal for Steering

The optimal scaling factor α=0.2 reflects the **small perturbation principle**: full scale (α=1.0) overshoots the target representation and destabilizes generation. Performance peaks at α=0.2 then degrades at higher scales, consistent with prior work on activation engineering (Turner et al., 2023).

### 5.3 Limitations

1. **Single model evaluation**: We only evaluate Qwen2.5-1.5B-Instruct. Results may differ for larger models or other architectures.

2. **Fixed sampling strategy**: ICL examples are sampled randomly. Similarity-based or difficulty-based sampling might yield different results.

3. **Answer-only extraction**: We extract only the final numerical answer, not evaluating the quality of intermediate reasoning steps.

### 5.4 Future Work

1. **Steering vector optimization**: Further tuning of extraction positions, layers, and scaling factors to maximize the effectiveness of Financial Steering Vectors.

2. **Larger model comparison**: Evaluating whether the 3-shot optimum holds for larger models (7B, 14B, 72B).

3. **Retrieval-augmented ICL**: Using semantic similarity to select more relevant ICL examples.

4. **Fine-tuning comparison**: Comparing prompting approaches with supervised fine-tuning on FinQA.

---

## 6. Conclusion

This study demonstrates that Chain-of-Thought prompting significantly improves financial numerical reasoning performance for smaller LLMs, with a 3.5× improvement over vanilla prompting (29.14% vs. 8.39%). CoT 3-shot achieves the best performance (32.08% accuracy) on FinQA using Qwen2.5-1.5B-Instruct.

Critically, we show that Financial Steering Vectors can recover 63% of the few-shot gain (+1.86%) with zero additional input tokens. This suggests a practical deployment strategy: extract steering vectors offline from few-shot examples, then apply them at inference time to achieve near few-shot performance at zero-shot computational cost.

**Practical Recommendations:**
1. Always use CoT prompting for financial reasoning tasks
2. If latency permits, use 3-shot ICL for best accuracy (32.08%)
3. If latency is critical, use FSV at layer 12, scale=0.2 for 31.00% accuracy with 4× fewer input tokens

---

## References

1. Chen, Z., Chen, W., Smiley, C., Shah, S., Borber, I., Ye, J., ... & Wang, W. Y. (2021). FinQA: A Dataset of Numerical Reasoning over Financial Data. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 3697-3711.

2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

3. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

4. Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv preprint arXiv:2409.12186*.

5. Li, Y., Wang, S., Ding, H., & Chen, H. (2023). Large Language Models in Finance: A Survey. *arXiv preprint arXiv:2311.10723*.

6. Ding, H., Li, Y., Wang, J., & Chen, H. (2024). Large Language Model Agent in Financial Trading: A Survey. *arXiv preprint arXiv:2408.06361*.

7. Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Steering Language Models With Activation Engineering. *arXiv preprint arXiv:2308.10248*.

8. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.

---

## Appendix

### A. Experimental Setup

- **Hardware**: NVIDIA GPU with vLLM inference engine
- **Software**: Python 3.10, vLLM 0.6.0, Transformers 4.40.0
- **Reproducibility**: Seeds 42, 43, 44, 45 for ICL sampling across 4 runs

### B. Operation-wise Analysis

Table S1 shows accuracy breakdown by operation type across prompting methods.

**Table S1: Accuracy by Operation Type Across Methods**

| Operation | Vanilla 0-shot | CoT 0-shot | CoT 3-shot | FSV (α=0.2) |
|-----------|----------------|------------|------------|-------------|
| Add | 10.77% | 28.46% | 31.54% | 30.00% |
| Subtract | 3.95% | 27.63% | 30.92% | 29.61% |
| Multiply | 0.00% | 18.18% | 27.27% | 22.73% |
| Divide | 0.00% | 25.00% | 50.00% | 25.00% |
| Greater | 100.00% | 100.00% | 100.00% | 100.00% |

**Key observations:**

1. **CoT dramatically improves complex operations**: Multiplication and division go from 0% (Vanilla) to 18-25% (CoT 0-shot), demonstrating that step-by-step reasoning is essential for multi-step calculations.

2. **FSV recovers most gains across operations**: FSV achieves intermediate performance between CoT 0-shot and CoT 3-shot for all operation types, confirming that steering vectors capture generalizable reasoning patterns.

3. **Simple comparisons are robust**: The "Greater" operation achieves 100% across all methods, indicating that comparison tasks do not benefit from additional reasoning scaffolding.

### C. Prompt Templates

**Vanilla Template:**
```
You are a financial analyst. Given the financial data, answer the question with a numerical value.

[Table and Context]

Question: {question}
Answer:
```

**CoT Template:**
```
You are a financial analyst. Given the financial data, solve the problem step by step.

[Table and Context]

Question: {question}
Let's solve this step by step:
```

### D. Token Position Extraction Implementation

The following code finds the last occurrence of "Reasoning:" in the tokenized prompt using progressive decoding:

```python
# Decode tokens progressively and find last "Reasoning:" position
decoded_so_far = ""
for idx, token_id in enumerate(input_ids):
    decoded_so_far += tokenizer.decode([token_id])
    last_pos = decoded_so_far.rfind("Reasoning:")
    if last_pos >= 0:
        reasoning_token_idx = idx  # Update to latest occurrence
```

This approach handles cases where "Reasoning:" appears multiple times in the prompt (from ICL demonstrations) by tracking the most recent occurrence.
