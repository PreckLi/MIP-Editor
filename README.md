# 🔗 MIP-Editor
Official implementation of the paper:  
**Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models**
Accepted at AAAI 2026 as a Conference Paper (Oral Presentation)

Pages of the main authors: [Kunhao Li](https://preckli.github.io/), [Di Wu](https://diwu.work/tagir-group/), [Lei Yang](https://www2.scut.edu.cn/sse/2018/0614/c16788a270682/page.htm), <br><br><br>
---

## 📌 Overview

**MIP-Editor** is a novel method for **cross-modal unlearning** in Multimodal Large Language Models (MLLMs). It identifies and edits influential neuron paths across vision and language modalities to selectively remove unwanted knowledge (e.g., memorized private data, harmful associations) without retraining the entire model.
![MIP-Editor](https://github.com/PreckLi/MIP-Editor/blob/main/pictures/mainfig.png)

### Abstract
Multimodal Large Language Models (MLLMs) extend foundation models to real-world applications by integrating inputs such as text and vision. However, their broad knowledge capacity raises growing concerns about privacy leakage, toxicity mitigation, and intellectual property violations. Machine Unlearning (MU) offers a practical solution by selectively forgetting targeted knowledge while preserving overall model utility. When applied to MLLMs, existing neuron-editing-based MU approaches face two fundamental challenges: (1) forgetting becomes inconsistent across modalities because existing point-wise attribution methods fail to capture the structured, layer-by-layer information flow that connects different modalities; and (2) general knowledge performance declines when sensitive neurons that also support important reasoning paths are pruned, as this disrupts the model’s ability to generalize. To alleviate these limitations, we propose a multimodal influential neuron path editor (MIP-Editor) for MU. Our approach introduces modality-specific attribution scores to identify influential neuron paths responsible for encoding forget-set knowledge and applies influential-path-aware neuron-editing via representation misdirection. This strategy also enables effective and coordinated forgetting across modalities while preserving the model's general capabilities. Experimental results demonstrate that MIP-Editor achieves a superior unlearning performance on multimodal tasks, with a maximum forgetting rate of 87.75% and up to 54.26% improvement in general knowledge retention. On textual tasks, MIP-Editor achieves up to 80.65% forgetting and preserves 77.9% of general performance.

## ⚙️ Run
To run the main pipeline with your own configurations of Multi-LLMs and Benchmarks:
```python main.py```

### 🧠 Influential Path Checkpoints
We provide precomputed influential neuron path checkpoints based on Qwen2.5-VL, generated on the MLLMU-Bench and CLEAR unlearning benchmarks.

🔗 Download Link:
[Baidu Netdisk](https://pan.baidu.com/s/1bosRsVY71rX-zQZv13ZS_g?pwd=8gc4)
🔑 Extraction Code: 8gc4
💡 These checkpoints contain the identified influential paths used by MIP-Editor for cross-modal unlearning. They can be directly loaded to reproduce our results without re-running path discovery.

Alternatively, you can regenerate the checkpoints from scratch by setting use_neuron_cache_flag = False in [main.py](main.py). This will recompute the influential paths during execution (note: this process may take several hours depending on your hardware).


## 📚 Citation
If you find this work useful in your research, please cite our paper:
```
@inproceedings{li2026crossmodal,
  title     = {Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models},
  author    = {Li, Kunhao and Li, Wenhao and Wu, Di and Yang, Lei and Bai, Jun and Jia, Ju and Xue, Jason},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```
