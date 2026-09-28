import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from text_classification.utils import saving_model
import copy
from pathlib import Path

def batch_train(model:torch.nn.Module,
                train_data_loader:torch.utils.data.DataLoader,
                val_data_loader:torch.utils.data.DataLoader,
                test_data_loader:torch.utils.data.DataLoader,
                epochs:int,
                divisor:int,
                device:torch.device,
                loss_curves:bool,
                optimiser:torch.optim,
                loss_fn:torch.nn,
                writer: torch.utils.tensorboard.SummaryWriter | None,
                scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                model_name:str,
                target_dir:str,
                accum_steps:int,
                grad_clip:float=1.0):
  """
  Function for training batches of data using dataloaders

  Each batch is iterative given to the function to be experimented
  on and then tested on.

  Args:
    model: The model being tested
    train_data_loader: The dataloader that contains the training data
    val_data_loader: The dataloader that contains the validating data
    test_data_loader: The dataloader that contains the testing data
    epochs: The amount of times the data is trained for
    divisor: Determines how much information is shown
    loss_curves: Determines if at the end loss curves are shown or not
    optimiser: Used for gradient descent
    loss_fn: Used for backpropagation and calculating loss
    writer: This tensorboard summary writer is for writing files and uploading
    them to tensorboard
    model_name: This name is used in the filepath for saving our model
    target_dir: The directory where our model is saved to
    vocab_size: The amount of words in the text
    accum_steps: The amount of epochs between optimiser updates
    grad_clip: The maximum allowed gradient

  Returns:
    Nothing
  """
  start = timer()
  patience = 20 # Increasing from 10 to 20 as the schedular will decrease the lr overtime
  scaler = torch.amp.GradScaler(device=device.type)
  overfit_counter = 0
  epochs_no_imp = 0
  best_model_loss = float("inf")
  best_stagnation_loss = float("inf")
  best_model_weights = copy.deepcopy(model.state_dict())

  results = {
      "train_loss": [],
      "val_loss": [],
      "train_acc": [],
      "val_acc": []
  }

  # if writer is not None:
  #  writer.add_graph(
  #    model=model,
  #    input_to_model=torch.randn(32,3,224,224).to(device)
  #  )

  for epoch in range(epochs):
      epoch_start = timer()
      print(f"Epoch: {epoch+1} \n ---------------------------------------------------------")
      train_loss, train_correct, train_total = 0.0,0,0          
      model.train()
      optimiser.zero_grad()
      for i, (x,y) in enumerate(train_data_loader):
          x,y = x.to(device), y.to(device)
          with torch.autocast(device_type=device.type, dtype=torch.float16):
            y_logits = model(x)
            loss = loss_fn(
              y_logits, y
            )
            train_loss += loss.item()
            loss = loss / accum_steps # Scaling loss so accumulated gradient match a large batch
            y_pred_labels = y_logits.argmax(dim=1)
            train_correct += (y_pred_labels == y).sum().item()
            train_total += y.size(0)
                      
          scaler.scale(loss).backward() # float.16 gradient values can become zero and vanish so we scale up the losses
          # and then do backpropagation
          is_last_batch = (i+1) == len(train_data_loader)
          # Optimiser is changed in batches of 4 so the if the dataloader is not divisible by 4 the remaining batch gradient are discarded
          # This allows the inclusion of the last remaining batches
          if(i+1) % accum_steps == 0 or is_last_batch:
            scaler.unscale_(optimiser)
            # we undo the scaling once done
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # prevents gradient weights from going above grad_clip and blowing up the weights
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad() # zeros the previous gradient so previous epochs don't contaminate the batch

          train_loss /= len(train_data_loader)

      val_loss, val_correct, val_total = 0.0,0,0
      model.eval()
      with torch.inference_mode():
          for x,y in val_data_loader:
              x,y = x.to(device), y.to(device)
              with torch.autocast(device_type=device.type, dtype=torch.float16):
                val_logits = model(x)
                loss = loss_fn(
                  val_logits, y
                )
                val_loss += loss.item()
                y_pred_labels = val_logits.argmax(dim=1)
                val_correct += (y_pred_labels == y).sum().item()
                val_total += y.size(0)

          val_loss /= len(val_data_loader)


      train_acc = (train_correct/train_total)*100
      val_acc = (val_correct/val_total)*100
      results["train_loss"].append(train_loss)
      results["val_loss"].append(val_loss)
      results["train_acc"].append(train_acc)
      results["val_acc"].append(val_acc)
      
      overfit_counter = detect_overfitting(results,epoch,overfit_counter)

      if val_loss < best_model_loss:
          best_model_loss = val_loss
          best_model_weights = copy.deepcopy(model.state_dict())

      best_stagnation_loss, epochs_no_imp = stagnation(val_loss,epoch,best_stagnation_loss, epochs_no_imp)
      if(epochs_no_imp >= patience):
          print("Loss is stagnant. Prematurely ending training!")
          break

      if writer is not None:
       writer.add_scalars(
           main_tag="Loss",
           tag_scalar_dict={
               "train_loss":train_loss,
               "val_loss":val_loss
           },
           global_step=epoch
       )

      epoch_time_elapsed = timer() - epoch_start
      if epoch % divisor == 0:
          print(f"\nTrain Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f} | Train Accuracy: {train_acc:.2f}% | Validation Accuracy: {val_acc:.2f}% | Time of Epoch: {epoch_time_elapsed:.2f}\n")

      if scheduler is not None:
          scheduler.step()
  

  model.load_state_dict(best_model_weights)
  model.eval()
  test_loss,test_correct, test_total = 0.0,0,0
  with torch.inference_mode():
       for x,y in test_data_loader:
            x,y = x.to(device),y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
              test_logits=model(x)
              test_loss = loss_fn(
                test_logits,y
              )
              y_pred_labels = test_logits.argmax(dim=1)
              test_correct += (y_pred_labels == y).sum().item()
              test_total += y.size(0)

  test_acc = (test_correct/test_total) * 100
  test_loss /= len(test_data_loader)
  print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")   
      
  if writer is not None: writer.close()
  end = timer()
  
  print(f"Training & Testing Finished! Total Training Time: {end-start:.2f} seconds")

  saving_model(
    model=model,
    model_name=model_name,
    target_dir=target_dir
  )
  if loss_curves:
    plot_loss_curves(results)

  training_finished()


def detect_overfitting(results,epoch:int,overfit_counter:int):

  """
  This helper function is used for detecting overfitting within the
  model that is being trained. Should the train and val loss become
  to different to each other then an overfitting warning is given

  Args:
    results: A dictionary contains keys of "train_loss", "train_acc",
    "val_loss" and "val_acc" each value is a list appended each epoch
    epoch: The amount of times the data is trained for
    overfit_counter: The amount of times the program has detected overfitting.

  Returns:
    overfit_counter: To be used by the program to see if the model
    is overfitting
  """

  min_delta = 0.01

  if(results["val_loss"][epoch] > results["val_loss"][epoch-1] + min_delta
     and results["train_loss"][epoch] < results["train_loss"][epoch-1] - min_delta):
      overfit_counter += 1
  else:
      overfit_counter = 0

  if overfit_counter > 3:
      print("Model is overfitting!")
  return overfit_counter


def stagnation(current_loss, epoch:int,best_loss, epochs_no_imp):

  """
  A helper function that detects if the model's training has become
  stagnant

  Args:
    current_loss: The validation loss
    epoch: The amount of times the data is trained for
    best_loss: The lowest loss the model has detected
    epochs_no_imp: The amount of epochs that the model has no improved for

  Returns:
    best_loss: The new best loss
    epochs_no_imp: The new best epochs_no_imp
  """

  min_delta = 1e-4
  if current_loss < best_loss - min_delta:
      best_loss = current_loss
      epochs_no_imp = 0
  else:
      epochs_no_imp += 1

  return best_loss, epochs_no_imp

def plot_loss_curves(results):
  """Plots training curves of a results dictionary.

  Args:
      results (dict): dictionary containing list of values, e.g.
          {"train_loss": [...],
            "val_loss": [...],
  """
  loss = results["train_loss"]
  val_loss = results["val_loss"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()


def training_finished():
  """
  Signals when training has concluded by implementing a beeping sound

  This function is designed for long training runs where the user may not
  be actively monitoring the process. It attempts to play a custom audio
  file (`done.wav`) when running in notebook environments such as Google
  Colab or Jupyter. If the audio playback is unavailable, it falls back
  to a system beep on compatible operating systems.

  Args:
    None

  Returns:
    None
  """
  sound_path = Path("assets/audio/done.wav")
  if sound_path.exists():
    try:
      from IPython.display import Audio, display
      display(Audio(str(sound_path), autoplay=True))
    except Exception:
       pass

  try:
    import winsound
    winsound.Beep(1000,1000)   
  except ImportError:
    pass