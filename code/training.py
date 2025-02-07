from building import *
import time
from tqdm import tqdm

PAD_TOKEN_ID = 0

def train(model, device, criterion, optimizer, dataloader, epochs, name: str):
    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for b, (src_batch, tgt_input_batch) in enumerate(dataloader):  
                # Move tensors to the correct device
                src_batch = src_batch.to(device)  # (B, S)
                tgt_input_batch = tgt_input_batch.to(device)  # (B, T)
                
                # Create tgt_output_batch by shifting tgt_input_batch left
                tgt_output_batch = tgt_input_batch[:, 1:].clone()  # Remove the first token
                pad_tensor = torch.full((tgt_output_batch.shape[0], 1), PAD_TOKEN_ID, device=device)
                tgt_output_batch = torch.cat([tgt_output_batch, pad_tensor], dim=1)  # Append PAD_TOKEN_ID at the end

                optimizer.zero_grad()  # Reset gradients

                # Forward pass
                logits = model(src_batch, tgt_input_batch)  # (B, T, vocab_size) - causal mask applied inside!!!

                # Reshape for loss computation -- note that criterion maps logits to tokens
                loss = criterion(logits.view(-1, model.vocab_size), tgt_output_batch.view(-1))

                # Backpropagation
                loss.backward()
                optimizer.step()
                
                # Track total loss
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save model to models directory
    model_save_path = f"../models/{name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
