# MVGCN-iSL

 THis is the code for our work "Multi-view graph convolutional network for predicting cancer cell-specific synthetic lethality". Our model, MVGCN-iSL, comprises three parts. In the first, the GCN processes multiple biological networks independently as cell-specific and cell-independent input graphs to obtain graph-specific representations that provide diverse information for SL prediction. In the second part, a max pooling operation integrates several graph-specific representations into one, and in the third part, a multi-layer deep neural network (DNN) model utilizes these integrated representations as input to predict SL.

## Requirements

We use `torch` as the architecture to build our deep learning model, and `torch-geometric` to implement graph neural network models. Here are a list of packages required to run our model:

- numpy
- pandas
- scipy
- scikit-learn
- networkx
- argparse
- tqdm
- torch
- torch-geometric

## Train the model

```
cd code/
python main.py --pooling attention --model GCN_attention --data_source A549 # training on A549 data
python main.py --pooling attention --model GCN_attention --data_source A549 --predict_novel_cellline 1 --novel_cellline Jurkat# test on Jurkat data
```
