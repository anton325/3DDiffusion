import torch


def interp(x,xp,fp):
    denom = (xp[:,1:] - xp[:,:-1])
    nom = (fp[1:] - fp[:-1])
    # nom = (fp[:,1:] - fp[:,:-1])
    m = nom / denom
    b = fp[:-1] - m * xp[:,:-1]
    idx = torch.sum(x >= xp,dim=-1) - 1
    idx = torch.clamp(idx, 0, m.shape[1] - 1)
    selected_bs = b[:,idx]
    selected_ms = m[:,idx]
    res = selected_ms * x + selected_bs
    max_values = torch.max(fp)
    min_values = torch.min(fp)
    clamped_tensor = torch.max(torch.min(res, max_values), min_values)
    return clamped_tensor

x = torch.tensor([1.5])
xp = torch.tensor([[0., 1.,2.,3.,3.]])
fp = (xp**2 + 1).reshape(-1)
i = interp(x,xp,fp)
print(i)

for x in range(0,20):
    x = torch.tensor([x/3])
    xp = 1+torch.tensor([[1., 2., 3.,4.,5.]])
    fp = (xp**2).squeeze(0)
    i = interp(x,xp,fp)
    print(i)