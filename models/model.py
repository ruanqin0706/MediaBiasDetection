import torch
import torch.nn as nn


class DetectionModel(nn.Module):
    def __init__(self, input_size=1024, max_len=200, num_filters=128, filter_size=[2, 3, 4, 5, 6]):
        super().__init__()
        self.layers_list = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=fs),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=num_filters, momentum=0.7),
            nn.MaxPool1d(max_len - fs)
        ) for fs in filter_size])
        self.linear_layer = nn.Sequential(nn.Linear(num_filters * len(filter_size), 1), nn.Sigmoid())

    def forward(self, inputs):
        # inputs: bs, embed_size, max_len

        out_list = [ll(inputs) for ll in self.layers_list]
        concatenated_tensor = torch.cat(out_list, dim=1)
        flatten = concatenated_tensor.squeeze(dim=-1)
        output = self.linear_layer(flatten)
        output = output.squeeze(dim=-1)

        return output


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)