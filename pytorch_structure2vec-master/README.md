# pytorch_structure2vec
pytorch implementation of structure2vec

## Setup

Build the c++ backend of s2v_lib and you are all set.

```
cd s2v_lib
make -j4  
```

## Reproduce Experiments on Harvard Clean Energy Project

First, you need to install rdkit (https://github.com/rdkit/rdkit) from source. Then set RDBASE to your built rdkit.
```
export RDBASE=/path/to/your/rdkit
```

Build the c++ backend of harvard_cep. 

```
cd harvard_cep
make -j4
```

### Prepare data

The raw data and cooked data are available at the following link:
https://www.dropbox.com/sh/eylta6a24fc9xo4/AAANyIgKnq49HB0Ud989JGEZa?dl=0

After you download the files, put them under the data folder. 

We used the same dataset in our paper (Dai. et.al, ICML 2016). Here the data split as is provided by [Wengong Jin](http://people.csail.mit.edu/wengong/) in [google drive](https://drive.google.com/drive/folders/0B0GLTTNiVPEkdmlac2tDSzBFVzg). So minor performance improvement is observed. 

##### cook data
The above dropbox folder already contains the cooked data. But if you want to cook it on your own, then you just need to download the raw txt data into the data folder, and do the following:

```
cd harvard_cep
python mol_lib.py
```

### Model dump

The pretrained model is under ```saved/``` folder. 

##### for mean_field: 
```
$ python main.py -gm mean_field -saved_model saved/mean_field.model -phase test
====== begin of s2v configuration ======
| msg_average = 0
======   end of s2v configuration ======
loading data
train: 1900000
valid: 82601
test: 220289
loading model from saved/epoch-best.model
loading graph from data/test.txt.bin
num_nodes: 6094162	num_edges: 7357400
100%|███████████████████████████████████████████████████████████████████████████████████| 220289/220289 [00:01<00:00, 130103.34it/s]
mae: 0.08846 rmse: 0.11290: 100%|███████████████████████████████████████████████████████████| 4406/4406 [00:15<00:00, 279.01batch/s]
average test loss: mae 0.07017 rmse 0.09724
```
##### for loopy_bp:
```
$ python main.py -gm loopy_bp -saved_model saved/loopy_bp.model -phase test
====== begin of s2v configuration ======
| msg_average = 0
======   end of s2v configuration ======
loading data
train: 1900000
valid: 82601
test: 220289
loading model from saved/loopy_bp.model
loading graph from data/test.txt.bin
num_nodes: 6094162	num_edges: 7357400
100%|███████████████████████████████████████████████████████████████████████████████████| 220289/220289 [00:01<00:00, 131913.93it/s]
mae: 0.06883 rmse: 0.08762: 100%|███████████████████████████████████████████████████████████| 4406/4406 [00:17<00:00, 246.84batch/s]
average test loss: mae 0.06212 rmse 0.08747

```

#### Reference

```bibtex
@article{dai2016discriminative,
  title={Discriminative Embeddings of Latent Variable Models for Structured Data},
  author={Dai, Hanjun and Dai, Bo and Song, Le},
  journal={arXiv preprint arXiv:1603.05629},
  year={2016}
}
```
