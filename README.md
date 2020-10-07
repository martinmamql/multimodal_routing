# multimodal_routing
Repository for EMNLP'20 paper Multimodal Routing: Improving Local and Global Interpretability of Multimodal Language Analysis.
[Arxiv]: https://arxiv.org/abs/2004.14198

### Prerequisites
- Python 3.6
- [Pytorch (>=1.2.0)](https://pytorch.org/) (performance might vary in different versions)
- CUDA 10.0 or above

### Datasets

Data files (containing processed MOSI, MOSEI and IEMOCAP datasets) can be downloaded from [here](https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0).

To retrieve the meta information and the raw data, please refer to the [SDK for these datasets](https://github.com/A2Zadeh/CMU-MultimodalSDK).

### Commands
To run the program which gives computational results including accuracy and F1 scores, 

```bash run.sh```

<!--- Analysis of interpretability in jupyter notebooks will come soon. --->

### Acknowledgement
Some portion of the code were adapted from the [multimodal_transformer](https://github.com/yaohungt/Multimodal-Transformer) repo.

