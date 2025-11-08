# MIP-Editor
Official implementation of the paper:  
**Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models**
Accepted at AAAI 2026 as a Conference Paper (Oral Presentation)

Pages of the main authors: [Kunhao Li](https://preckli.github.io/), [Di Wu](https://diwu.work/tagir-group/), [Lei Yang](https://www2.scut.edu.cn/sse/2018/0614/c16788a270682/page.htm), <br><br><br>
---

## 📌 Overview

**MIP-Editor** is a novel method for **cross-modal unlearning** in Multimodal Large Language Models (MLLMs). It identifies and edits influential neuron paths across vision and language modalities to selectively remove unwanted knowledge (e.g., memorized private data, harmful associations) without retraining the entire model.
![MIP-Editor](https://github.com/PreckLi/MIP-Editor/blob/main/pictures/mainfig.png)

## Run
To run the main pipeline:
```python main.py```

📚 Citation
If you find this work useful in your research, please cite our paper:
```
@inproceedings{li2026crossmodal,
  title     = {Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models},
  author    = {Li, Kunhao and Li, Wenhao and Wu, Di and Yang, Lei and Bai, Jun and Jia, Ju and Xue, Jason},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```
