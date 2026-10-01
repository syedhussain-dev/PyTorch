import os
from pathlib import Path
import zipfile
import requests
import torch
import random
from typing import Tuple, List, Dict
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

def saving_model(model:torch.nn.Module, model_name:str, target_dir:str):
  """
  This helper functions is for saving an exact replica of the model we are training

  This is done by saving it to a target directory that if doesn't exist will be
  created. Making sure the model ends with the appropriate extension. Finally,
  using torch.save to save it

  Args:
    model: The model we are trying to save
    model_name: The name we give to the model when saving
    target_dir: The target directory we want to save the
    model to

  Returns:
    None
  """
  model_path = Path(target_dir)
  if not model_path.exists():
    model_path.mkdir(parents=True, exist_ok=True)
  assert model_name.endswith((".pt", ".pth")), "model_name should end with '.pt' or '.pth'"
  model_save_path = model_path / model_name
  print(f"Saving model to {model_save_path}")
  torch.save(obj=model.state_dict(),f=model_save_path)

def loading_model(model_class,model_save_path, device,*args,**kwargs):

  """
  This helper function is for loading models of any type that were
  previously saved

  This is done by creating a new instance of a class and using
  torch.load to save the model to that instance then moving it
  to the target device and evaluating it

  Args:
    model_class: The class used to create that model
    model_save_path: The filepath where the model is saved
    device: The device the model ran on
    *args, **kwargs: Any additional arguments used by the class

  Returns:
    loadel_model: It returns the model that we loaded
  """
  if model_class is not None:
    loaded_model = model_class(*args,**kwargs)
  loaded_model.load_state_dict(torch.load(f=model_save_path, map_location=device))
  loaded_model.to(device)
  loaded_model.eval()
  return loaded_model

def set_seeds(seed: int=42):
  """Sets random sets for torch operations.

  Args:
      seed (int, optional): Random seed to set. Defaults to 42.
  """
  # Set the seed for general torch operations
  torch.manual_seed(seed)
  # Set the seed for CUDA torch operations (ones that happen on the GPU)
  torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
  """Downloads a zipped dataset from source and unzips to destination.

  Args:
      source (str): A link to a zipped file containing data.
      destination (str): A target directory to unzip data to.
      remove_source (bool): Whether to remove the source after downloading and extracting.
  
  Returns:
      pathlib.Path to downloaded data.
  
  Example usage:
      download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                    destination="pizza_steak_sushi")
  """
  # Setup path to data folder
  data_path = Path("data/")
  data_path.mkdir(parents=True, exist_ok=True)

  zip_file = data_path / "watch_shoe_fragrance.zip"
  extract_path = data_path / destination

  # download only if needed
  if not extract_path.exists():
    print("Downloading data zip")
    request = requests.get(source)

    with open(zip_file, "wb") as f:
        f.write(request.content)

    print("Unzipping")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_path)

    if remove_source:
      os.remove(zip_file)
    
    return extract_path
    
def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None):
  """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

  log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

  Where timestamp is the current date in YYYY-MM-DD format.

  Args:
      experiment_name (str): Name of experiment.
      model_name (str): Name of model.
      extra (str, optional): Anything extra to add to the directory. Defaults to None.

  Returns:
      torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

  Example usage:
      # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
      writer = create_writer(experiment_name="data_10_percent",
                              model_name="effnetb2",
                              extra="5_epochs")
      # The above is the same as:
      writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
  """
  from datetime import datetime
  import os

  # Get timestamp of current date (all experiments on certain day live in same folder)
  timestamp = datetime.now().strftime("%d-%m-%Y") 


  log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
  if extra: log_dir = os.path.join(log_dir, extra)
      
  print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
  return SummaryWriter(log_dir=log_dir)

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
  dir_path (str): target directory

  Returns:
  A print out of:
    number of subdiretories in dir_path
    number of images (files) in each subdirectory
    name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
      print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def model_size_and_params(model, model_savepath) -> Dict:
  model_size = Path(model_savepath).stat().st_size // (1024**2)
  total_params = sum(torch.numel(param) for param in model.parameters())
  return {"MODEL_SIZE":model_size,"TOTAL_PARAMETERS":total_params}

def can_fit_batch(model: torch.nn.Module, input_shape, batch_size: int, device: torch.device):
    model = model.to(device)
    model.train()

    try:
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)

        x = torch.randint(
            0,
            model.vocab_size,
            (batch_size, *input_shape),
            device=device,
            dtype=torch.long
        )

        output = model(x)
        loss = output.mean()
        loss.backward()

        del x, output, loss
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False
        raise

def find_max_batch_size(
    model,
    input_shape,
    device,
    start=1,
    max_batch=2048,
):
    """
    Finds the maximum batch size that fits using binary search.
    """
    low = start
    high = start
    while high <= max_batch and can_fit_batch(model, input_shape, high, device):
        low = high
        high *= 2
    high = min(high, max_batch)
    while low + 1 < high:
        mid = (low + high) // 2
        if can_fit_batch(model, input_shape, mid, device):
            low = mid
        else:
            high = mid
    return low