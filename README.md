PyTorch DGCNN
=============

Updates 6/13/2018
-----------------
Switch to PyTorch 0.4.0 now. Please update your PyTorch version.

Updates 4/16/2018
-----------------

Added support to multi-dimensional continuous node features. Added support to use your own datasets. 

About
-----

PyTorch implementation of DGCNN (Deep Graph Convolutional Neural Network). Check https://github.com/muhanzhang/DGCNN for more information.

Installation
------------

This implementation is based on Hanjun Dai's PyTorch version of structure2vec. Please first unzip the pytorch_structure2vec-master.zip by

    unzip pytorch_structure2vec-master.zip

Then, under the "pytorch_structure2vec-master/s2vlib/" directory, type

    make -j4

to build the necessary c++ backend.

After that, under the root directory of this repository, type

    ./run_DGCNN.sh

to run DGCNN on dataset DD with default settings.

Or type 

    ./run_DGCNN.sh DATANAME FOLD

to run on dataset = DATANAME using fold number = FOLD (1-10, corresponds to which fold to use as test data in the cross-validation experiments).

If you set FOLD = 0, e.g., typing "./run_DGCNN.sh DD 0", then it will run 10-fold cross validation on DD and report the average accuracy.

Alternatively, type

    ./run_DGCNN.sh DATANAME 1 200

to use the last 200 graphs in the dataset as testing graphs. The fold number 1 will be ignored.

Check "run_DGCNN.sh" for more options.

Datasets
--------

Default graph datasets are stored in "data/DSName/DSName.txt". Check the "data/README.md" for the format. 

In addition to the support of discrete node labels (tags), DGCNN now supports multi-dimensional continuous node features. One example dataset with continuous node features is "Synthie". Check "data/Synthie/Synthie.txt" for the format. 

There are two preprocessing scripts in MATLAB: "mat2txt.m" transforms .mat graphs (from Weisfeiler-Lehman Graph Kernel Toolbox), "dortmund2txt.m" transforms graph benchmark datasets downloaded from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

How to use your own data
------------------------

The first step is to transform your graphs to the format described in "data/README.md". You should put your testing graphs at the end of the file. Then, there is an option -test_number X, which enables using the last X graphs from the file as testing graphs. You may also pass X as the third argument to "run_DGCNN.sh" by

    ./run_DGCNN.sh DATANAME 1 X

where the fold number 1 will be ignored.

Reference
---------

If you find the code useful, please cite our paper:

    @inproceedings{zhang2018end,
      title={An End-to-End Deep Learning Architecture for Graph Classification.},
      author={Zhang, Muhan and Cui, Zhicheng and Neumann, Marion and Chen, Yixin},
      booktitle={AAAI},
      year={2018}
    }

Muhan Zhang, muhan@wustl.edu
3/19/2018
