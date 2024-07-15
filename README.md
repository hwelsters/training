Wassump homie it's Tony.  
Once you clone this repo, there's a bit more you gotta do:  
Download the model here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  
This should be put into the root.  

Make a venv. Then run pip install -r requirements.txt.  

In the venv, run this first, to make sure you're using CUDA  
```python a.py```
If it says false, run dis.
```pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

Try to run train.py ig:  
```python train.py```