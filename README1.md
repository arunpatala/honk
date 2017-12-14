# Honk: CNNs for Keyword Spotting

Honk is a PyTorch reimplementation of Google's TensorFlow convolutional neural networks for keyword spotting, which accompanies the recent release of their [Speech Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html). For more details, please consult our writeup:

+ Raphael Tang, Jimmy Lin. [Honk: A PyTorch Reimplementation of Convolutional Neural Networks for Keyword Spotting.](https://arxiv.org/abs/1710.06554) _arXiv:1710.06554_, October 2017.

Honk is useful for building on-device speech recognition capabilities for interactive intelligent agents. Our code can be used to identify simple commands (e.g., "stop" and "go") and be adapted to detect custom "command triggers" (e.g., "Hey Siri!").

- If you do not have PyTorch, please see [the website](http://pytorch.org).
- Install Python dependencies: `pip install -r requirements.txt`
- Fetch the data and models: `./fetch_data.sh`

If you need to adjust options, like turning off CUDA, please edit `config.json`.

In our [honk-models](https://github.com/castorini/honk-models) repository, there are several pre-trained models for Caffe2 (ONNX) and PyTorch. The `fetch_data.sh` script fetches these models and extracts them to the `model` directory. You may specify which model and backend to use in the config file's `model_path` and `backend`, respectively. Specifically, `backend` can be either `caffe2` or `pytorch`, depending on what format `model_path` is in. Note that, in order to run our ONNX models, the packages `onnx` and `onnx_caffe2` must be present on your system; these are absent in requirements.txt.

### Training and evaluating the model
**CNN models**. `python -m utils.train --mode [train|eval]` trains or evaluates the model. It expects all training examples to follow the same format as that of [Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). The recommended workflow is to download the dataset and add custom keywords, since the dataset already contains many useful audio samples and background noise.

**Residual models**. We recommend the following hyperparameters for training any of our `res{8,15,26}[-narrow]` models on the Speech Commands Dataset:
```
python -m utils.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 26 --weight_decay 0.00001 --lr 0.1 0.01 0.001 --schedule 3000 6000 --model res{8,15,26}[-narrow]
```
For more information about our deep residual models, please see our paper:

+ Raphael Tang, Jimmy Lin. [Deep Residual Learning for Small-Footprint Keyword Spotting.](https://arxiv.org/abs/1710.10361) _arXiv:1710.10361_, October 2017.

There are command options available:

| option         | input format | default | description |
|----------------|--------------|---------|-------------|
| `--batch_size`   | [1, n)       | 100     | the mini-batch size to use            |
| `--cache_size`   | [0, inf)       | 32768     | number of items in audio cache, consumes around 32 KB * n   |
| `--conv1_pool`   | [1, inf) [1, inf) | 2 2     | the width and height of the pool filter       |
| `--conv1_size`     | [1, inf) [1, inf) | 10 4  | the width and height of the conv filter            |
| `--conv1_stride`     | [1, inf) [1, inf) | 1 1  | the width and length of the stride            |
| `--conv2_pool`   | [1, inf) [1, inf) |  1 1    | the width and height of the pool filter            |
| `--conv2_size`     | [1, inf) [1, inf) | 10 4  | the width and height of the conv filter            |
| `--conv2_stride`     | [1, inf) [1, inf) | 1 1  | the width and length of the stride            |
| `--data_folder`   | string       | /data/speech_dataset     | path to data       |
| `--dev_every`    | [1, inf)     |  10     | dev interval in terms of epochs            |
| `--dev_pct`   | [0, 100]       | 10     | percentage of total set to use for dev        |
| `--dropout_prob` | [0.0, 1.0)   | 0.5     | the dropout rate to use            |
| `--gpu_no`     | [-1, n] | 1  | the gpu to use            |
| `--group_speakers_by_id` | {true, false} | true | whether to group speakers across train/dev/test |
| `--input_file`   | string       |      | the path to the model to load   |
| `--input_length`   | [1, inf)       | 16000     | the length of the audio   |
| `--lr`           | (0.0, inf)   | {0.1, 0.001}   | the learning rate to use            |
| `--mode`         | {train, eval}| train   | the mode to use            |
| `--model`        | string       | cnn-trad-pool2 | one of `cnn-trad-pool2`, `cnn-tstride-{2,4,8}`, `cnn-tpool{2,3}`, `cnn-one-fpool3`, `cnn-one-fstride{4,8}`, `res{8,15,26}[-narrow]`, `cnn-trad-fpool3`, `cnn-one-stride1` |
| `--momentum` | [0.0, 1.0) | 0.9 | the momentum to use for SGD |
| `--n_dct_filters`| [1, inf)     | 40      | the number of DCT bases to use  |
| `--n_epochs`     | [0, inf) | 500  | number of epochs            |
| `--n_feature_maps` | [1, inf) | {19, 45} | the number of feature maps to use for the residual architecture |
| `--n_feature_maps1` | [1, inf)             | 64        | the number of feature maps for conv net 1            |
| `--n_feature_maps2`   | [1, inf)       | 64     | the number of feature maps for conv net 2        |
| `--n_labels`   | [1, n)       | 4     | the number of labels to use            |
| `--n_layers` | [1, inf) | {6, 13, 24} | the number of convolution layers for the residual architecture |
| `--n_mels`       | [1, inf)     |   40    | the number of Mel filters to use            |
| `--no_cuda`      | switch     | false   | whether to use CUDA            |
| `--noise_prob`     | [0.0, 1.0] | 0.8  | the probability of mixing with noise    |
| `--output_file`   | string    | model/google-speech-dataset.pt     | the file to save the model to        |
| `--seed`   | (inf, inf)       | 0     | the seed to use        |
| `--silence_prob`     | [0.0, 1.0] | 0.1  | the probability of picking silence    |
| `--test_pct`   | [0, 100]       | 10     | percentage of total set to use for testing       |
| `--timeshift_ms`| [0, inf)       | 100    | time in milliseconds to shift the audio randomly |
| `--train_pct`   | [0, 100]       | 80     | percentage of total set to use for training       |
| `--unknown_prob`     | [0.0, 1.0] | 0.1  | the probability of picking an unknown word    |
| `--wanted_words` | string1 string2 ... stringn  | command random  | the desired target words            |


