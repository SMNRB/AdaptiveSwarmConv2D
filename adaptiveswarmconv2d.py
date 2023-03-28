import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSwarmConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_swarms=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AdaptiveSwarmConv2D, self).__init__()

        self.num_swarms = num_swarms
        self.swarms = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias) for _ in range(num_swarms)
        ])
        self.adaptation_net = nn.Sequential(
            nn.Linear(in_channels * kernel_size * kernel_size, in_channels * kernel_size * kernel_size),
            nn.ReLU(),
            nn.Linear(in_channels * kernel_size * kernel_size, in_channels * kernel_size * kernel_size)
        )
        self.attention = nn.MultiheadAttention(out_channels, num_heads=8)

    def adapt_individual_filters(self, swarm):
        filters = swarm.weight.view(swarm.weight.size(0), -1)
        adapted_filters = self.adaptation_net(filters)
        adapted_filters = adapted_filters.view_as(swarm.weight)
        return adapted_filters

    def coordinate_swarm(self, x, swarm):
        filters = self.adapt_individual_filters(swarm)
        swarm_out = F.conv2d(x, filters, swarm.bias, swarm.stride, swarm.padding, swarm.dilation, swarm.groups)
        attn_out, _ = self.attention(swarm_out.flatten(2).permute(2, 0, 1), swarm_out.flatten(2).permute(2, 0, 1), swarm_out.flatten(2).permute(2, 0, 1))
        attn_out = attn_out.permute(1, 2, 0).view_as(swarm_out)
        return attn_out

    def forward(self, x):
        swarm_outputs = [self.coordinate_swarm(x, swarm) for swarm in self.swarms]
        combined_out = torch.cat(swarm_outputs, dim=1)
        return combined_out
