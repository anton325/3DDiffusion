import torch


import torch

def torch_interp(x,xp,fp):
    denom = (xp[1:] - xp[:-1])
    nom = (fp[1:] - fp[:-1])
    # nom = (fp[:,1:] - fp[:,:-1])
    m = nom / denom
    
    isinf = torch.isinf(m)
    
    # Use cumsum to find the first invalid in each row
    invalid_cumsum = isinf.cumsum(dim=0)
    
    # Find the last valid values in each row before the first invalid
    last_valid_values = torch.where(invalid_cumsum == 0, m, torch.tensor(float('nan')).to(m.device))
    last_valid_values = torch.nan_to_num(last_valid_values[-1:], nan=0, posinf=0, neginf=0)  # Handle rows with all invalids
    
    # Broadcast the last valid values to the invalid positions
    m = torch.where(isinf, last_valid_values, m)

    b = fp[:-1] - m * xp[:-1]
    idx = torch.sum(x.reshape(x.shape[0],1) >= xp,dim=-1) - 1
    idx = torch.clamp(idx, 0, m.shape[0] - 1)
    selected_bs = b[idx] # b[:,idx]
    selected_ms = m[idx]
    res = selected_ms * x + selected_bs
    max_values = torch.max(fp)
    min_values = torch.min(fp)
    clamped_tensor = torch.max(torch.min(res, max_values), min_values)
    return clamped_tensor

x = torch.tensor([1.5])
xp = torch.tensor([0., 1.,2.,3.,3.])
fp = (xp**2 + 1).reshape(-1)
i = torch_interp(x,xp,fp)
print(i)

for x in range(0,20):
    x = torch.tensor([x/3])
    xp = 1+torch.tensor([1., 2., 3.,4.,5.])
    fp = xp**2
    i = torch_interp(x,xp,fp)
    print(i)