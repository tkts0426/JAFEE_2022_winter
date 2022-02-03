import torch

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, exp_path, cond_path, target_path):
        self.exp_path = exp_path
        self.cond_path = cond_path
        self.target_path = target_path
        self.inputs = []
        self.targets = []
        for i in range(exp_path.shape[0]):
            self.inputs.append([exp_path[i, :], cond_path[i, :]])
            self.targets.append(target_path[i, :])

    def __len__(self):
        return self.exp_path.shape[0]

    def __getitem__(self, idx):
        out_input = self.inputs[idx]
        out_target =  self.targets[idx]

        return out_input, out_target