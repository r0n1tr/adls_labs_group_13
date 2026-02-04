# Advanced Deep Learning Systems Report - Group 13
---

## Overview

This document summarises the work completed across all **ADLS labs and tutorials**, following the official MASE / ADLS workflow.  
The focus of the labs was on **model optimisation, hardware–software co-design, and deployment-aware deep learning**, using the MASE toolchain.

---

## Lab 0 — Introduction to MASE

## Tutorial 1

This tutorial introduces the Mase framework, explaining how it imports, represents, and transforms neural network models for hardware and software optimisation.

### Key Concepts


#### 1. Importing Models

- Demonstrates how to import standard PyTorch models (BERT from HuggingFace) into the Mase environment.
- Mase uses Torch FX to "trace" the PyTorch code and capture a compute graph, converting Python code into a structured graph of nodes.

#### 2. MaseGraph (The Intermediate Representation)

- MaseGraph is the central data structure, acting as a wrapper around the Torch FX graph.
- Node Types: The graph consists of six atomic node types: placeholder (inputs), call_module (layers like Linear/Conv), call_function (torch ops like torch.add), call_method, get_attr (params), and output.
- Mase Operators: Mase adds an abstraction layer over raw PyTorch ops (e.g., module_related_func, builtin_func) to help unify software and hardware definitions.

#### 3. The Pass System

- Mase relies on a compiler-style "pass" system to optimise graphs.
- Analysis Passes: Read-only operations that annotate the graph with data (e.g., calculating tensor shapes, inferring data types).
- Transform Passes: Operations that modify the graph topology (e.g., deleting layers or inserting logic).
- Standard Interface: Every pass follows the signature: pass(mg, args) -> (mg, outputs).

#### 4. Practical Examples

- Metadata Initialisation: Using dummy inputs to propagate shapes through the network (init_metadata_analysis_pass, add_common_metadata_analysis_pass).
- Custom Analysis: Writing a simple function to count the number of Dropout layers.
- Custom Transformation: Writing a function to remove Dropout layers for inference optimisation.
Exporting & Checkpointing

The tutorial concludes by showing how to save the transformed graph to disk (mg.export()) and reload it later (MaseGraph.from_checkpoint()).


Qna Questions:

- Why Torch FX? Mase chooses FX over ONNX or TorchScript because it is Python-native. It allows graph transformations using pure Python code without needing a C++ runtime or complex external standard.
- The Necessity of Dummy Inputs: In the "Shape Propagation" step, Mase requires a real dummy input tensor to run a forward pass. This is how it determines specific tensor sizes (H, W, Channels) at every node in the graph.
- Graph Integrity: When writing a Transform Pass to remove a node (like Dropout), you cannot just delete it. You must use node.replace_all_uses_with(parent_node) first. If you delete a node that is still being used as an input by another node, the graph becomes invalid and will crash.
- Where is the data stored? All Mase-specific information (quantisation bits, hardware settings, shapes) is stored in a hidden dictionary on each node: node.meta["mase"].

## Tutorial 2

This tutorial uses a pre-trained BERT model for sentiment analysis on the IMDb dataset, comparing standard Supervised Fine-Tuning (SFT) against the LoRA (Low-Rank Adaptation) technique.

### Key Concepts

#### 1. Task Setup

- Goal: Classify movie reviews as positive or negative using the IMDb dataset.
- Model: A "tiny" variant of BERT is loaded from HuggingFace.
- Architecture Adjustment: A classification head (Linear Layer + Cross Entropy Loss) is attached to the pre-trained encoder.

#### 2. Customising the Compute Graph

- Demonstrates how MaseGraph can be customised to include specific inputs like labels (which triggers loss calculation) and `attention_mask`.
- By removing unused inputs (defaults like `token_type_ids`), the graph can be simplified.

#### 3. Supervised Fine-Tuning (SFT)

- Training Strategy: We freeze the largest part of the model (Embeddings) and train all other weights (Encoder layers + Classifier head).
- Cost: While effective, SFT requires updating a large number of parameters (order of millions), consuming significant memory.

#### 4. Parameter Efficient Fine-Tuning (PEFT) with LoRA
- Concept: Instead of updating the massive weight matrix *W*, LoRA updates two tiny low-rank matrices *A* and *B*, such that  
  *W*<sub>new</sub> = *W*<sub>fixed</sub> + *A* × *B*.
- Mase Implementation Using `insert_lora_adapter_transform_pass`, the standard Linear layers are automatically swapped for LoRA layers.
- Benefit: Reduces trainable parameters significantly (approx. **4.5× reduction** in this example), speeding up training and reducing memory usage with comparable accuracy.

#### 5. Optimising for Inference*
- Fusion: After training, the `fuse_lora_weights_transform_pass` permanently merges the learned *A* × *B* update into the original weights *W*.
- Result: The model returns to its original architecture (*Y* = *XW*<sub>fused</sub> + *b*), ensuring inference is just as fast as the original model with no extra computational overhead.


#### Important Details for Q&A
- Why freeze embeddings? In NLP models, the embedding layer often contains the vast majority of parameters but requires little adjustment for downstream tasks. Freezing it drastically saves resources.
- LoRA Trade-off: The "rank" (r) parameter controls the balance. Higher rank = more capacity to learn (higher accuracy) but more memory. Lower rank = extreme efficiency.
- Inference Speed: A LoRA model during training is slower (extra matrix multiplications). A LoRA model after fusion is identical in speed to the original base model.
- Mase Pass: The fuse_lora_weights_transform_pass is crucial for deployment; without it, you are running extra operations for no reason.
---

## Lab 1 — Model Compression

### Objectives
- Apply compression techniques to reduce model size
- Evaluate accuracy–efficiency trade-offs

### Work Completed

### Observations

### Key Takeaway

---

## Lab 2 — Neural Architecture Search (NAS)

### Objectives
- Explore automated architecture optimisation
- Define and evaluate search spaces

### Work Completed



![Alt text](./lab2.2.png)


### Observations

### Key Takeaway
---

## Lab 3 — Mixed Precision Optimisation

### Objectives
- Investigate mixed-precision inference
- Balance numerical precision and performance

### Work Completed

**Task 1:** Modified the Optuna search to allow per-layer quantisation hyperparameters for `LinearInteger`. Each layer can independently select width $\in \{8, 16, 32\}$ and fractional width $\in \{2, 4, 8\}$, exposing these as searchable hyperparameters via `trial.suggest_categorical()`.

**Task 2:** Extended the search space to include all supported quantised layer types: `LinearInteger`, `LinearMinifloatIEEE`, `LinearBlockFP`, `LinearBlockLog`, `LinearLog`, and `LinearBinary` (`LinearBlockMinifloat` was not tested due to an issue in Mase). Each layer type required specific configuration parameters (exponent widths, block sizes, etc.) to be passed correctly.

We then ran 50 trials using Optuna's `RandomSampler` on `BERT-tiny` for IMDb sentiment classification.

![Maximum Accuracy vs Trials](task1_accuracy_vs_trials.png)

![Accuracy by Layer Type](task2_accuracy_by_precision.png)

### Observations

- All quantisation methods except Binary converge to ~85% accuracy, matching full-precision FP32 performance.
- `LinearBinary` (1-bit) starts poorly (~64%) but reaches 82.9% with the right configuration, only 2.6% below baseline while achieving **32x compression**.
- `LinearLog` slightly outperformed other quantised formats, achieving both the highest peak (85.8%) and mean accuracy (83.6%). This suggests logarithmic representation better captures the weight distribution of transformer models, which typically have many small values with some larger outliers.
- The best overall model (85.79%) used **mixed precision**, combining different layer types: LinearLog for feed-forward layers, LinearBinary for less-sensitive attention outputs, and FP32 for critical attention projections.

### Key Takeaway
Moderate quantisation (8-16 bit) provides essentially free compression with negligible accuracy loss. For aggressive compression, binary quantisation offers a compelling trade-off: 32x smaller models with only ~3% accuracy drop, ideal for edge & IoT deployment where memory and compute are constrained. Mixed-precision search reveals that different layers have different quantisation sensitivities, and the optimal configuration uses the right precision for each layer rather than a uniform approach.

---

## Lab 4 — Hardware & Software Co-Design

### Hardware Stream

#### Objectives
- Emit hardware-oriented representations
- Prepare models for acceleration

### Work Completed

### Observations

### Key Takeaway
---

## Overall Reflection

The ADLS labs provided hands-on experience with **end-to-end deep learning system design**, bridging the gap between:
- Model development
- Optimisation techniques
- Deployment constraints
- Hardware considerations

The MASE framework enabled a structured and realistic workflow aligned with modern industry practices.

---

## Tools & Technologies

- MASE Toolchain  
- PyTorch  
- Model Compression Techniques  
- Neural Architecture Search  
- Mixed-Precision Inference  

---

## Conclusion

Through these labs, I developed a deeper understanding of **deployment-aware machine learning**, reinforcing the importance of system-level thinking when designing and optimising modern deep learning models.
