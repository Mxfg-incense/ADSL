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

The following command is an example to train the model on the A549 cell line with the transformer pooling method. 

```bash
cd code/
python main.py --balanced 0 --data_source A549  --pooling transformer --epoch 200 --neg_num 2 --test 0 --esm_reps_flag 1 --MLP_celline 1
```
After training, the checkpoint will be saved in the `ckpt/` folder and test results and experiment logs will be saved in the `results/` folder.
To load the trained model and evaluate it on the test set, you can use the following command:

```bash
python main.py --balanced 0 --data_source A549  --pooling transformer --epoch 200 --neg_num 2 --test 1 --esm_reps_flag 1 --MLP_celline 1
```