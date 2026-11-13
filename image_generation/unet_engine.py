import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import copy

from image_generation.utils import saving_model
from models.StableDiffusionV1.DDIMScheduler import DDIMScheduler
def batch_train(
    unet:torch.nn.Module,
    vae:torch.nn.Module,
    clip:torch.nn.Module,
    ddim_scheduler:DDIMScheduler,
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    test_dataloader:DataLoader,
    device:torch.device,
    precision:torch.dtype,
    loss_fn:torch.nn.Module,
    optimiser:torch.optim.Optimizer,
    epochs:int,
    divisor:int,
    accum_steps:int,
    model_name:str,
    max_seq_len:int,
    target_dir:str="saved_models",
    loss_curves:bool=False,
    scheduler:torch.optim.lr_scheduler.LRScheduler|None=None,
):
    train_start = timer()
    best_model_loss = float('inf')
    scalar = torch.amp.GradScaler(
        device=device.type,
        enabled=(device.type == "cuda" and precision == torch.float16)
    )
    patience = 20
    grad_clip = 1.0
    min_delta = 1e-4
    epochs_no_imp = 0
    overfit_counter = 0

    vae.eval()
    clip.eval()
    vae.requires_grad_(False)
    clip.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    results = {
        "train_loss":[],
        "val_loss":[]
    }

    for epoch in range(epochs):
        epoch_start = timer()
        print(f"Epoch: {epoch+1} \n--------------------------------------------------------------------------------------- ")
        train_loss = 0.0
        unet.train()
        optimiser.zero_grad()
        for batch,(image,caption) in enumerate(train_dataloader):
            image = image.to(device)
            
            tokens = tokenizer(
                text=caption,
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                return_tensors="pt"
            )
            token_ids = tokens["input_ids"].to(device)
            with torch.no_grad():
                latent_image = vae(image)
                context = clip(token_ids)
            noise = torch.randn_like(latent_image)
            timestep = torch.randint(low=0,high=1000,size=(latent_image.shape[0],),device=device,dtype=torch.long)
            noisy_latent = ddim_scheduler.add_noise(latent_image,noise,timestep)

            with torch.autocast(device_type=device.type,dtype=precision, enabled=(precision != "float32")):
                pred_noise = unet(noisy_latent,context,timestep)
                loss = loss_fn(
                    pred_noise.float(),noise.float()
                )
                train_loss += loss.item()
                loss /= accum_steps
            scalar.scale(loss).backward()
            is_last_batch = True if (batch+1) == len(train_dataloader) else False
            if(batch+1) % accum_steps == 0 or is_last_batch:
                scalar.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(unet.parameters(),grad_clip)
                scalar.step(optimiser)
                scalar.update()
                optimiser.zero_grad()


        train_loss /= len(train_dataloader)

        val_loss = 0.0
        unet.eval()
        with torch.inference_mode():
            for batch,(image,caption) in enumerate(val_dataloader):
                image = image.to(device)
                            
                tokens = tokenizer(
                    text=caption,
                    padding="max_length",
                    max_length=max_seq_len,
                    truncation=True,
                    return_tensors="pt"
                )
                token_ids = tokens["input_ids"].to(device)
                with torch.no_grad():
                    latent_image = vae(image)
                    context = clip(token_ids)
                noise = torch.randn_like(latent_image)
                timestep = torch.randint(low=0,high=1000,size=(latent_image.shape[0],),device=device,dtype=torch.long)
                noisy_latent = ddim_scheduler.add_noise(latent_image,noise,timestep)
    
                with torch.autocast(device_type=device.type,dtype=precision, enabled=(precision != "float32")):
                    pred_noise = unet(noisy_latent,context,timestep)
                    loss = loss_fn(
                        pred_noise.float(), noise.float()
                    )
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
            
        epochs_no_imp = detect_stagnation(results,epoch,epochs_no_imp,min_delta)
        if(epochs_no_imp > patience):
            print("Model is no longer improving. Terminating training...")
            break

        overfit_counter = detect_overfitting(results,overfit_counter,epoch,min_delta)

        if val_loss < best_model_loss:
            best_model_loss = val_loss
            best_model_weights = copy.deepcopy(unet.state_dict())

        epoch_end = timer()
        if epoch % divisor == 0:
            print(f"\nTrain Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f} | Time of Epoch: {time_diff(epoch_start,epoch_end):.2f}\n")

        if scheduler is not None:
            scheduler.step()    

    unet.load_state_dict(best_model_weights)
    unet.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for batch,(image,caption) in enumerate(test_dataloader):
            image = image.to(device)
                        
            tokens = tokenizer(
                text=caption,
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                return_tensors="pt"
            )
            token_ids = tokens["input_ids"].to(device)
            with torch.no_grad():
                latent_image = vae(image)
                context = clip(token_ids)
            noise = torch.randn_like(latent_image)
            timestep = torch.randint(low=0,high=1000,size=(latent_image.shape[0],),device=device,dtype=torch.long)
            noisy_latent = ddim_scheduler.add_noise(latent_image,noise,timestep)

            with torch.autocast(device_type=device.type,dtype=precision, enabled=(precision != "float32")):
                pred_noise = unet(noisy_latent,context,timestep)
                loss = loss_fn(
                    pred_noise.float(), noise.float()
                )
                test_loss += loss.item()
        test_loss /= len(test_dataloader)

    print(f"Test Loss: {test_loss:.5f}")
    train_end = timer()
    print(f"Training & Testing Finished! Total Training Time: {time_diff(train_start,train_end):.2f}")
    saving_model(
        unet,model_name,target_dir
    )
    if loss_curves:
        plot_loss_curves(results)


def detect_overfitting(results:dict[str,list],overfit_counter:int,epoch:int,min_delta:float):
    if epoch == 0: return overfit_counter
    if(results["train_loss"][epoch] < results["train_loss"][epoch-1] + min_delta and 
       results["val_loss"][epoch] > results["val_loss"][epoch-1] - min_delta):
        overfit_counter += 1
    else:
        overfit_counter = 0

    if overfit_counter >= 3:
        print("Model is overfitting!")
    return overfit_counter

def detect_stagnation(results:dict[str,list],epoch:int,epochs_no_imp:int,min_delta:float):
    if epoch == 0: return epochs_no_imp
    if(abs(results["val_loss"][epoch] - results["val_loss"][epoch-1]) < min_delta):
        epochs_no_imp += 1
    else:
        epochs_no_imp = 0
    return epochs_no_imp

def time_diff(start:float,end:float):
    return end - start
    
def plot_loss_curves(results:dict[str,list]):
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]

    epochs = range(len(train_loss))

    plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)
    plt.plot(epochs,train_loss,label="train_loss")
    plt.plot(epochs,val_loss,label="validation_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()