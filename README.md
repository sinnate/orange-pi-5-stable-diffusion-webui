# How to install Stable diffusion webui on a Orange pi 5

I'm using Armbian with Rockship patch, you can also use RebornOS or any Orange pi offical image 


```bash
# Dependancy
sudo apt install wget git python3 python3-venv python3-dev
```

```bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```
After the webui.sh failed to install torch and torchvision we need to install it into the virtual envoriment to do that we need to switch to the virtual envoriment and then install them with pip
```bash
source venv/bin/source
pip install torch torchvision
```
and then run
```bash
./webui.sh --skip-torch-cuda-test  
```
