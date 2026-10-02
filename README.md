# Faithful Grounded Visual Reasoning via Learned Proxy-Tokens

<div align="center">

#### 2026 IEEE International Conference on Image Processing (ICIP)

### [**Paper**](https://ieeexplore.ieee.org/document/11630456) &nbsp;&nbsp;|&nbsp;&nbsp; [**Hugging Face**](https://huggingface.co/tomhodemon)

</div>

## Abstract

Multimodal Large Language Models (MLLMs) have achieved remarkable success in Visual Question Answering (VQA), yet their "black-box" nature hinders deployment in critical domains. Grounded Visual Reasoning (GVR) approaches attempt to improve interpretability by explicitly couple textual rationales with visual grounding information, which are typically textual coordinates. This mechanism lacks a learnable semantic link to the visual features, often resulting in a semantic-spatial gap where the model hallucinates coordinates that do not correspond to image evidences. In this work, we introduce Composer, a MLLM that leverages a novel visual grounding mechanism based on learned proxy-tokens to promote faithful interpretability. These discrete symbolic pointers explicitly index the image latent space, allowing the model to manipulate visual regions as addressable, semantically manipulable sets. To rigorously validate our novel grounding mechanism, we constructed ComposerGCoT, a dataset synthesized to enable holistic assessment of reasoning consistency and grounding accuracy. Experimental results indicate that Composer achieves performance parity with its coordinate-based counterpart in final answer accuracy, while improving visual grounding accuracy by +9.0 points. By demonstrating that discrete proxy-tokens capture spatial semantics more effectively than typical textual coordinates, we establish that visual grounding mechanisms with learnable semantic links represent a promising path toward trustworthy and reliable MLLMs.

## Citation

If you find this work helpful for your research, please consider citing us:

```bibtex
@INPROCEEDINGS{11630456,
  author={Hodemon, Tom and Chaouch, Mohamed and Tuo, Aboubacar and Loesch, Angelique},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)}, 
  title={Faithful Grounded Visual Reasoning Via Learned Proxy-Tokens}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Cognition;Cognitive systems;Modeling;Visualization;Grounding;Printing;Accuracy;Learning (artificial intelligence);Joining processes;Large language models;MLLMs;Grounded Visual Reasoning;Interpretability},
  doi={10.1109/ICIP61757.2026.11630456}
}
```

This codebase is built upon: [**Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models**](https://github.com/FoundationVision/Groma). We thanks them for their amazing work!