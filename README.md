# How to Install Stable Diffusion WebUI on an Orange Pi 5

This guide outlines the steps to install Stable Diffusion WebUI on an Orange Pi 5.
Note : I'm currently looking for a way to make RK NPU to work with it.

Reduce your resolution, it will take at least 2-6min for a step to complete.

I'm using the `Orange pi 5 8G model` with a 8G swap, for better performance i recommend nvme swap for at least 8G more.

## Dependencies

The following dependencies need to be installed before proceeding:
```bash
sudo apt install wget git python3 python3-venv python3-dev
```
## Installation

To install Stable Diffusion WebUI, run the following command:
```bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```
Note that after running `webui.sh` , the installation of torch and torchvision will fail. we will need to install them into the virtual environment manually. To do so, switch to the virtual environment and install them using pip:
```bash
cd stable-diffusion-webui
source venv/bin/activate
pip install torch torchvision
```
After installing torch and torchvision, run the following command to launch, you also add `--listen` to access it from your local network.
```bash
./webui.sh --skip-torch-cuda-test --lowvram --no-half --opt-sdp-attention --use-cpu all
```
In order to use Stable Diffusion WebUI, you will need to download Stable Diffusion models. One option is to download a pre-trained model from Hugging Face. For example, you can download the "openjourney" model by visiting https://huggingface.co/prompthero/openjourney.

After downloading the model, you will need to place the .ckpt or .safetensors files into the models/Stable-diffusion directory.

This should complete the installation process for Stable Diffusion WebUI on your Orange Pi 5.
