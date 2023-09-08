import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize

import src.utils as utils
from src.dataloader import DatasetSegmentation

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam

dataset_path = "./bottle_glass_dataset"

sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
sam_lora = LoRA_sam(sam,4)  
processor = Samprocessor(sam_lora.sam)
dataset = DatasetSegmentation(dataset_path, processor)
#utils.plot_image_mask_dataset(dataset, 3)

train_dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
optimizer = Adam(sam_lora.lora_vit.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_lora.sam
model.to(device)
model.train()


for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      inv_batch = utils.dict_list_inversion(batch)
      outputs = model(batched_input=inv_batch,
            multimask_output=False)
         
      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())


    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')