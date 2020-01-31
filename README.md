# [Drill-down: Interactive Retrieval of Complex Scenes using Natural Language Queries.](https://arxiv.org/abs/1911.03826)
Fuwen Tan, Paola Cascante-Bonilla, Xiaoxiao Guo, Hui Wu, Song Feng, Vicente Ordonez. NeurIPS 2019


## Overview
This paper explores the task of interactive image retrieval using natural language queries, where a user progressively provides input queries to refine a set of retrieval results. Moreover, our work explores this problem in the context of complex image scenes containing multiple objects. We propose Drill-down, an effective framework for encoding multiple queries with an efficient compact state representation that significantly extends current methods for single-round image retrieval. We show that using multiple rounds of natural language queries as input can be surprisingly effective to find arbitrarily specific images of complex scenes. Furthermore, we find that existing image datasets with textual captions can provide a surprisingly effective form of weak supervision for this task. We compare our method with existing sequential encoding and embedding networks, demonstrating superior performance on two proposed benchmarks: automatic image retrieval on a simulated scenario that uses region captions as queries, and interactive image retrieval using real queries from human evaluators.

## Requirements
- Setup a conda environment and install some prerequisite packages like this
```bash
conda create -n retrieval python=3.6    # Create a virtual environment
source activate retrieval         	    # Activate virtual environment
conda install jupyter scikit-image cython opencv seaborn nltk pycairo h5py  # Install dependencies
python -m nltk.downloader all				    # Install NLTK data
```
- Please also install [pytorch](http://pytorch.org/) 1.0 (or higher), torchVision, and torchtext


## Data 
- Download the images of the Visual Genome dataset if you have not done so
```Shell
./experiments/scripts/fetch_images.sh
```
This will populate the `DrillDown/data` folder with `vg/VG_100K` and `vg/VG_100K_2`.

- Download the annotations of the images
```Shell
./experiments/scripts/fetch_annotations.sh
```
This will populate the `DrillDown/data` folder with `vg/sg_xmls` and `vg/rg_jsons`, which are per-image scene-graph and region-graph annotations.

- Download the global image features and region features
```Shell
./experiments/scripts/fetch_features.sh
```
This will populate the `DrillDown/data` folder with `vg/global_features` and `vg/region_36_final`, which are the global features and region features of the images. The global features were extracted from a pretrained ResNet101 model. The region features were extracted from a pretrained FasterRCNN model provided by https://github.com/peteanderson80/bottom-up-attention. Please see `tools/save_image_features.py`, the FasterRCNN repo, and `tools/save_region_features.py` for more details.

- Download the pretrained models
```Shell
./experiments/scripts/fetch_pretrained_models.sh
```
This will populate the `DrillDown/data` folder with `caches/image_ckpts` and `caches/region_ckpts`


## Training/evaluation scripts
The training/evaluation scripts of different models are also included in `./experiments/scripts`. 
The results will appear in `DrillDown/logs`
Please note that, when finetuning the supervisedly pretrained DrillDown model, e.g. runing the script
```Shell
./experiments/scripts/train_drill_down_3x128_reinforce.sh
```
the default pretrained model is `DrillDown/caches/region_ckpts/vg_f128_i3_sl_ckpt.pkl`.



## Citing

If you find our paper/code useful, please consider citing:

	@InProceedings{drilldown,
    author={Fuwen Tan and Paola Cascante-Bonilla and Xiaoxiao Guo and Hui Wu and Song Feng and Vicente Ordonez},
    title={Drill-down: Interactive Retrieval of Complex Scenes using Natural Language Queries},
    booktitle = {Neural Information Processing Systems (NeurIPS)},
    month = {December},
    year = {2019}
    }


    
# License

This project is licensed under the [MIT license](https://opensource.org/licenses/MIT):

Copyright (c) 2019 University of Virginia, Fuwen Tan, Paola Cascante-Bonilla, Xiaoxiao Guo, Hui Wu, Song Feng, Vicente Ordonez.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.







