import torch
from diffusions.diffusions import DiffusionModel
from models.models import DiT
import torch
from dataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import logging
from pathlib import Path

train_name = '256-6-1767-normal-600less-4000-ref-split-1'

Path('training_weight',(train_name)).mkdir(exist_ok=True)

logging.basicConfig(filename=f'log/{train_name}.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

best_score, score, epochs, early_stop_time, early_stop_threshold = 1e10, 0, 200, 0, 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dit_model = DiT(hidden_size=256, depth=6, num_heads=16, mlp_ratio=4.0, esm_feature_size=1280, max_protein_length=600).to(device)
dm = DiffusionModel(eps_model=dit_model, n_steps=1000, device=device)  
optimizer = torch.optim.Adam(dm.eps_model.parameters(), lr=1e-4)

with open("dataset/dataset_1767_4000_nano_normal_600less.pkl", 'rb') as f:
    data = pickle.load(f)
dataset = ProteinDataset(data)
dataset_size = len(dataset)
print(dataset_size)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_indices = train_dataset.indices
test_indices = test_dataset.indices

logging.info(f'Train dataset size: {len(train_dataset)}, Indices: {train_indices}')
logging.info(f'Test dataset size: {len(test_dataset)}, Indices: {test_indices}')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for epoch in range(epochs):
    loss_record = []
    dm.eps_model.train()
    for step, batch in enumerate(train_loader):
        esm = batch["ESM"].squeeze(0).float().to(device)  # [1, 1, 1280]
        ref_1 = batch["REFCONF1"].squeeze(0).float().to(device)
        ref_2 = batch["REFCONF2"].squeeze(0).float().to(device)
        optimizer.zero_grad()
        for i in range(0, 4000, 400):
            md_chunk = batch["MD"].squeeze(0).float()[i:i + 100].to(device)  # [1, chunk_size, protein_length, 3]
            loss = dm.loss(md_chunk, esm, ref_1, ref_2)  
            loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

    
    mean_train_loss = torch.tensor(loss_record).mean().item()
    print(f'Training epoch: {epoch}, mean loss: {mean_train_loss}')
    logging.info(f'Epoch {epoch}, Training Loss: {mean_train_loss}')

    loss_record = []
    dm.eps_model.eval()  
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            frame_indices = torch.randint(0, 4000, (100,), device=device)
            md = batch["MD"].squeeze(0).float().to(device)[frame_indices] # [batch_size, num_conformations, protein_length, 3]
            esm = batch["ESM"].squeeze(0).float().to(device)  # [batch_size, 1, 1280]
            ref_1 = batch["REFCONF1"].squeeze(0).float().to(device)
            ref_2 = batch["REFCONF2"].squeeze(0).float().to(device)
            loss = dm.loss(md, esm, ref_1, ref_2)
            loss_record.append(loss.item())
    
    mean_val_loss = torch.tensor(loss_record).mean().item()
    logging.info(f'Epoch {epoch}, Validation Loss: {mean_val_loss}')

    if (epoch + 1) % 20 == 0:
        model_save_path = f'training_weight/{train_name}/model_epoch_{epoch+1}.pth'
        torch.save(dm.eps_model.state_dict(), model_save_path)
        logging.info(f'Epoch {epoch+1}, Model saved at {model_save_path}')
        print(f'Model saved at {model_save_path}')

    if mean_val_loss < best_score:
        early_stop_time = 0
        best_score = mean_val_loss
        torch.save(dm.eps_model.state_dict(), f'training_weight/{train_name}/best_model.pth')  
        logging.info(f'Epoch {epoch}, Model saved with Validation Loss: {mean_val_loss}')
    else:
        early_stop_time += 1

    if early_stop_time > early_stop_threshold:
        print("Early stopping triggered.")
        logging.info("Early stopping triggered.")
        break

    print(f'early_stop_time/early_stop_threshold: {early_stop_time}/{early_stop_threshold}, mean val loss: {mean_val_loss}')
    logging.info(f'Epoch {epoch}, Early Stop Time: {early_stop_time}/{early_stop_threshold}')
