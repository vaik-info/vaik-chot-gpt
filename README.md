# vaik-chot-gpt

Useful when you want to take experiment easy. Train and inference with the LLM. 


## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-chot-gpt
```

## Usage

### Train

```python
import os

from vaik_chot_gpt.train_seq2seq import train

train_input_file_path = os.path.expanduser('~/.vaik-mnist-text-dataset/dataset/mnist_train.txt')
test_input_file_path = os.path.expanduser('~/.vaik-mnist-text-dataset/dataset/mnist_valid.txt')
test_output_dir_path = os.path.expanduser('~/.vaik-chot-gpt/mnist_model/')
model_name = 't5-small'
num_train_epochs = 10
per_device_train_batch_size = 4
learning_rate = 3e-5
weight_decay = 0.01

train(train_input_file_path, test_input_file_path, test_output_dir_path, model_name, num_train_epochs,
      per_device_train_batch_size, learning_rate, weight_decay)

```

### Inference

```shell
import os

from vaik_chot_gpt.inference import inference

input_model_dir_path = os.path.expanduser('~/.vaik-chot-gpt/mnist_model/model')
test_input_file_path = os.path.expanduser('~/.vaik-mnist-text-dataset/dataset/mnist_valid.txt')
model_name = 't5-small'

inference(input_model_dir_path, test_input_file_path, model_name)
#input_text: [[  0   0 , output_text: 6, pred_output_text: 1
#input_text: [[  0   0 , output_text: 0, pred_output_text: 4
#input_text: [[  0   0 , output_text: 4, pred_output_text: 4
#ACC:0.74
```