# AdaptiveSwarmConv2D

AdaptiveSwarmConv2D is a novel PyTorch module for computer vision tasks that combines the concept of swarm intelligence with adaptive filters to learn more effectively from the input data. The module is designed to work with standard PyTorch pipelines and can be used as a drop-in replacement for conventional convolutional layers.

## Features

### Multiple swarms of filters
Instead of using a fixed set of filters, AdaptiveSwarmConv2D employs multiple swarms of filters. Each swarm is responsible for capturing different features within the input data, and swarms work together to create a more comprehensive representation of the input.

### Adaptive filter shapes and sizes
The filters within each swarm are not fixed in shape and size. They can adapt their shape and size during training to better capture the relevant features in the input data. This adaptive behavior makes the AdaptiveSwarmConv2D module more flexible and capable of learning complex patterns.

### Self-attention mechanism
A self-attention mechanism is used for swarms to identify important regions within the input. This allows the swarms to focus on the most relevant parts of the input data and allocate more resources to learning features from those areas.

### Compatibility with standard PyTorch pipelines
AdaptiveSwarmConv2D is designed to be compatible with standard PyTorch pipelines, which means it can be used in place of conventional convolutional layers with minimal modifications to existing code.

## How it works

The AdaptiveSwarmConv2D module uses multiple swarms of filters, with each swarm being responsible for learning a different aspect of the input data. Each filter within a swarm can adapt its shape and size during training to better capture the relevant features in the input data.

A self-attention mechanism is employed to allow the swarms to focus on the most important regions of the input data. This attention mechanism is based on the input feature maps and helps the swarms to allocate more resources to learning features from the most relevant areas of the input.

The output feature maps from each swarm are combined to create a comprehensive representation of the input data. This representation is then passed through the rest of the network for further processing.

## Usage 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptiveswarmconv2d import AdaptiveSwarmConv2D

class CustomModel(nn.Module):
    def __init__(self, ...):
        super(CustomModel, self).__init__()
        self.conv1 = AdaptiveSwarmConv2D(in_channels, out_channels, kernel_size, num_swarms, ...)
        ...

    def forward(self, x):
        x = self.conv1(x)
        ...
        return x
```

## Example 

Here is an example of using the AdaptiveSwarmConv2D module in a simple neural network model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptiveswarmconv2d import AdaptiveSwarmConv2D

class Znn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout=0.5, num_swarms=3):
        super(Znn, self).__init__()
        self.conv1 = AdaptiveSwarmConv2D(in_channels, 32, kernel_size=3, num_swarms=num_swarms)
        self.conv2 = AdaptiveSwarmConv2D(32 * num_swarms, 64, kernel_size=3, num_swarms=num_swarms)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(64 * num_swarms * 5 * 5, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        x = x.view(-1, 64 * self.conv2.num_swarms * 5 * 5)
        
        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
```

## Contributing

We welcome contributions to improve this project. Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
