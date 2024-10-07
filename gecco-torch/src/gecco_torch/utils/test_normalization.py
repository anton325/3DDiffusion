import torch.nn.functional as F
import torch
# source https://github.com/Lyken17/GroupNorm.pytorch/blob/master/group_norm.py
def custom_group_norm(x,eps,num_groups):
    # print(x.size())
    N, C, H = x.size()
    G = num_groups
    # assert C % G == 0

    x = x.reshape(N, G, -1)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)

    x = (x - mean) / (var + eps).sqrt()
    x = x.view(N, C, H)
    return x


num_groups = 16
eps = 1e-5
inputs = torch.randn(20, 32,10)
inputs[19,31,:] = 0
print(inputs.shape)
print(inputs[19,31,:])
gn = F.group_norm(inputs, num_groups, weight = None, bias = None, eps=eps)
gn_custom = custom_group_norm(inputs,eps,num_groups)

diffs = []
percents = []
for x in range(gn.shape[0]):
    for y in range(gn.shape[1]):
        for z in range(gn.shape[2]):
            diff = abs(gn[x][y][z] - gn_custom[x][y][z])
            diffs.append(diff)
            if gn[x][y][z] > gn_custom[x][y][z]:
                # print(gn[x][y][z]/gn_custom[x][y][z]-1)
                percents.append(gn[x][y][z]/gn_custom[x][y][z]-1)
            else:
                # print(gn_custom[x][y][z]/gn[x][y][z]-1)
                percents.append(gn_custom[x][y][z]/gn[x][y][z]-1)
            # print(inputs[x][y][z],gn[x][y][z],gn_custom[x][y][z])
print(sum(diffs)/len(diffs)) # -> 0.0204 daneben im durchschnitt
print(sum(percents)/len(percents)) # -> 0.05% daneben im durchschnitt

print(gn_custom[19,31,:])
