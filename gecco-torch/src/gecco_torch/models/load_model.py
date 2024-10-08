import shutil
from pathlib import Path
import torch
from gecco_torch.structs import Mode

def load_model(version,epoch=None):
    if len(version)>6:
        # on cluster
        path_python_main = f"/home/giese/claix_work/gecco_logs/version_{version}/gaussians_conditional.py"
        if not Path(path_python_main).exists():
            raise Exception(f"{path_python_main} existiert nicht, wahrscheinlich müssen die cluster ordner gemounted werden")
    else:
        # local
        path_python_main = f"/globalwork/giese/gecco_shapenet/logs/version_{version}/gaussians_conditional.py"
    print(f"copy {path_python_main}")
    shutil.copy(path_python_main,"/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/models/temp.py")
    from gecco_torch.models.temp import model, render_fn, data, mode, reparam
    print(f"loaded from model {path_python_main} mode {mode}")

    checkpoint_path = Path(Path(path_python_main).parent,"checkpoints")
    selection = None
    best_val = 1000
    path_to_best = None
    for checkpoints in Path(checkpoint_path).glob("*.ckpt"):
        if "continued" in checkpoints.name:
            continue
        
        if epoch is not None and f"epoch={epoch}" in checkpoints.name:
            selection = checkpoints
            break

        if epoch is None:
            val = float(checkpoints.name.split("=")[-1].split(".")[0])
            if val < best_val:
                best_val = val
                path_to_best = checkpoints

    if epoch is None:
        selection = path_to_best
    epoch = int(selection.name.split("=")[1].split("-")[0])
    print(f"Load checkpoint {selection}")
    checkpoint_state_dict = torch.load(selection, map_location='cuda')
    model_state_dict = checkpoint_state_dict['state_dict']
    # model_state_dict = checkpoint_state_dict['ema_state_dict']

    # Load the state dict
    model.load_state_dict(model_state_dict)
    return model, render_fn, data, mode, reparam, epoch



def load_model_saver_checkpoints(version, working_dir):
    #  always assume that it's local
    
    file_name = "gaussians_unconditional"
    path_python_main = f"/globalwork/giese/gecco_shapenet/logs/version_{version}/{file_name}.py"
    # es ist conditional
    unconditional_bool = True # es ist unconditional

    if not Path(path_python_main).exists():
        unconditional_bool = False # es ist conditional, weil das conditional existiert
        print(f"{path_python_main} existiert nicht, wahrscheinlich ist es unconditional")
        file_name = "gaussians_conditional"
        path_python_main = f"/globalwork/giese/gecco_shapenet/logs/version_{version}/{file_name}.py"

    if not Path(path_python_main).exists():
        raise Exception(f"{path_python_main} existiert nicht, weder unconditional noch conditional")


    print(f"copy {path_python_main}")

    shutil.copy(path_python_main,Path(working_dir,"gecco-torch","src","gecco_torch","models","temp.py"))
    from gecco_torch.models.temp import model, render_fn, data, mode, reparam

    checkpoint_path = Path(Path(path_python_main).parent,"saver_checkpoints")
    for checkpoints in Path(checkpoint_path).glob("*"):
        selection = checkpoints
    epoch = int(selection.name.split("_")[1].split("_")[0])
    print(f"Load checkpoint {selection}")
    model_state_dict = torch.load(selection, map_location='cuda')
    # Load the state dict
    try:
        model.load_state_dict(model_state_dict)
    except:
        if Mode.dino_triplane in mode:
            # ich habe die umbeannt 
            model_state_dict['backbone.model.triplane_conv_xy.weight'] = model_state_dict['backbone.model.triplane_xy.weight'].clone()
            del model_state_dict['backbone.model.triplane_xy.weight']
            model_state_dict['backbone.model.triplane_conv_xy.bias'] = model_state_dict['backbone.model.triplane_xy.bias'].clone()
            del model_state_dict['backbone.model.triplane_xy.bias']
            model_state_dict['backbone.model.triplane_conv_yz.weight'] = model_state_dict['backbone.model.triplane_yz.weight'].clone()
            del model_state_dict['backbone.model.triplane_yz.weight']
            model_state_dict['backbone.model.triplane_conv_yz.bias'] = model_state_dict['backbone.model.triplane_yz.bias'].clone()
            del model_state_dict['backbone.model.triplane_yz.bias']
            model_state_dict['backbone.model.triplane_conv_xz.weight'] = model_state_dict['backbone.model.triplane_xz.weight'].clone()
            del model_state_dict['backbone.model.triplane_xz.weight']
            model_state_dict['backbone.model.triplane_conv_xz.bias'] = model_state_dict['backbone.model.triplane_xz.bias'].clone()
            del model_state_dict['backbone.model.triplane_xz.bias']
        model.load_state_dict(model_state_dict)
    # paar sachen müssen besonders gehandhabt werden

    return model, render_fn, data, mode, reparam, epoch, unconditional_bool


def load_model_saver_checkpoints_old(version):
    
    file_name = "gaussians_unconditional"
    if len(version)>6:
        # on cluster
        path_python_main = f"/home/giese/claix_work/gecco_logs/version_{version}/{file_name}.py"
        if not Path(path_python_main).exists():
            path_python_main = f"/home/giese/claix_hpcwork/gecco_logs/version_{version}/{file_name}.py"
            if not Path(path_python_main).exists():
                print(f"{path_python_main} existiert nicht, wahrscheinlich müssen die cluster ordner gemounted werden")
    else:
        # local
        path_python_main = f"/globalwork/giese/gecco_shapenet/logs/version_{version}/{file_name}.py"
    # es ist conditional
    unconditional_bool = True # es ist unconditional

    if not Path(path_python_main).exists():
        unconditional_bool = False # es ist conditional, weil das conditional existiert
        print(f"{path_python_main} existiert nicht, wahrscheinlich ist es unconditional")
        file_name = "gaussians_conditional"
        if len(version)>6:
            # on cluster
            path_python_main = f"/home/giese/claix_work/gecco_logs/version_{version}/{file_name}.py"
            if not Path(path_python_main).exists():
                path_python_main = f"/home/giese/claix_hpcwork/gecco_logs/version_{version}/{file_name}.py"
                if not Path(path_python_main).exists():
                    raise Exception(f"{path_python_main} existiert nicht, wahrscheinlich müssen die cluster ordner gemounted werden")
        else:
            # local
            path_python_main = f"/globalwork/giese/gecco_shapenet/logs/version_{version}/{file_name}.py"
    
    if not Path(path_python_main).exists():
        raise Exception(f"{path_python_main} existiert nicht, weder unconditional noch conditional")


    print(f"copy {path_python_main}")
    shutil.copy(path_python_main,"/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/models/temp.py")
    from gecco_torch.models.temp import model, render_fn, data, mode, reparam

    checkpoint_path = Path(Path(path_python_main).parent,"saver_checkpoints")
    for checkpoints in Path(checkpoint_path).glob("*"):
        selection = checkpoints
    epoch = int(selection.name.split("_")[1].split("_")[0])
    print(f"Load checkpoint {selection}")
    model_state_dict = torch.load(selection, map_location='cuda')
    # Load the state dict
    try:
        model.load_state_dict(model_state_dict)
    except:
        if Mode.dino_triplane in mode:
            model_state_dict['backbone.model.triplane_conv_xy.weight'] = model_state_dict['backbone.model.triplane_xy.weight'].clone()
            del model_state_dict['backbone.model.triplane_xy.weight']
            model_state_dict['backbone.model.triplane_conv_xy.bias'] = model_state_dict['backbone.model.triplane_xy.bias'].clone()
            del model_state_dict['backbone.model.triplane_xy.bias']
            model_state_dict['backbone.model.triplane_conv_yz.weight'] = model_state_dict['backbone.model.triplane_yz.weight'].clone()
            del model_state_dict['backbone.model.triplane_yz.weight']
            model_state_dict['backbone.model.triplane_conv_yz.bias'] = model_state_dict['backbone.model.triplane_yz.bias'].clone()
            del model_state_dict['backbone.model.triplane_yz.bias']
            model_state_dict['backbone.model.triplane_conv_xz.weight'] = model_state_dict['backbone.model.triplane_xz.weight'].clone()
            del model_state_dict['backbone.model.triplane_xz.weight']
            model_state_dict['backbone.model.triplane_conv_xz.bias'] = model_state_dict['backbone.model.triplane_xz.bias'].clone()
            del model_state_dict['backbone.model.triplane_xz.bias']
        model.load_state_dict(model_state_dict)
    # paar sachen müssen besonders gehandhabt werden

    return model, render_fn, data, mode, reparam, epoch, unconditional_bool

if __name__ == "__main__":
    epoch = 1
    version = "450705"
    # load_model(version)
    load_model_saver_checkpoints(version)

