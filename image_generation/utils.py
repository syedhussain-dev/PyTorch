import torch
from pathlib import Path
import os

def saving_model(model:torch.nn.Module, model_name:str, target_dir:str):
    """
    This function is for saving a model

    Model path is created if it does not exist. An assert is added to see if the model
    name has the proper file extension. Models parameters are stored in the file.

    Args:
        model: The model being saved
        model_name: The name of the file to which the parameters of the model are saved
        target_dir: The directory where the model is saved

    Returns:
        model_path: The full path of the file
    """
    model_path = Path(target_dir)
    if not model_path.exists():
        os.mkdir(model_path)
    assert model_name.endswith(".pt",".pth"), "Model name must have necessary file extension"
    model_path = model_path / model_name
    print(f"Saving model to {model_path}")
    torch.save(obj=model.state_dict(),f=model_path)
    return model_path

def loading_model(model_class,model_path:str,device:torch.device,*args,**kwargs):
    """
    This function is for loading a model

    Model class is specified and an object is instantiated to load the models parameters.
    The model is then moved to the target device, evaluated to turn off gradients and returned.

    Args:
        model_class: The class of the model we're attempting to load
        model_path: File where the models saved parameters are contained
        device: The target device (device agnostic code)
        *args, **kwargs: Any additional arguements that the model class may need

    Returns:
        loaded_model: The model containing the parameters
    """
    if model_class is not None:
        loaded_model = model_class(*args,**kwargs)
    loaded_model = torch.load(f=model_path,map_location=device)
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model