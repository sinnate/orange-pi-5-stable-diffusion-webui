#!/usr/bin/env python
# coding: utf-8

from typing import List
from rknn.api import RKNN
from math import exp
from sys import exit
import argparse


def convert_pipeline_component(onnx_path: str, resolution_list: List[List[int]], target_platform: str = 'rk3588'):
    print(f'Converting {onnx_path} to RKNN model')
    print(f'with target platform {target_platform}')
    print(f'with resolutions:')
    for res in resolution_list:
        print(f'- {res[0]}x{res[1]}')
    use_dynamic_shape = False
    if(len(resolution_list) > 1):
        print("Warning: RKNN dynamic shape support is probably broken, may throw errors")
        use_dynamic_shape = True

    batch_size = 1
    LATENT_RESIZE_FACTOR = 8
    # build shape list
    if "text_encoder" in onnx_path:
        input_size_list = [[[1,77]]]
        inputs=['input_ids']
        use_dynamic_shape = False
    elif "unet" in onnx_path:
        # batch_size = 2  # for classifier free guidance # broken for rknn python api

        input_size_list = []
        for res in resolution_list:
            input_size_list.append(
                [[1,4, res[0]//LATENT_RESIZE_FACTOR, res[1]//LATENT_RESIZE_FACTOR],
                 [1],
                 [1, 77, 768],
                 [1, 256]]
            )
        inputs=['sample','timestep','encoder_hidden_states','timestep_cond']
    elif "vae_decoder" in onnx_path:
        input_size_list = []
        for res in resolution_list:
            input_size_list.append(
                [[1,4, res[0]//LATENT_RESIZE_FACTOR, res[1]//LATENT_RESIZE_FACTOR]]
            )
        inputs=['latent_sample']
    else:
        print("Unknown component: ", onnx_path)
        exit(1)

    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # , single_core_mode=True
    rknn.config(target_platform='rk3588', optimization_level=3,
                dynamic_input= input_size_list if use_dynamic_shape else None, disable_rules=['convert_layernorm_to_exnorm'] )
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_path,
                         inputs=None if use_dynamic_shape else inputs,
                         input_size_list= None if use_dynamic_shape else input_size_list[0])   
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, rknn_batch_size=batch_size)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    #export
    print('--> Export RKNN model')
    ret = rknn.export_rknn(onnx_path.replace('.onnx', '.rknn'))
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    rknn.release()
    print('RKNN model is converted successfully!')


def parse_resolution_list(resolution: str) -> List[List[int]]:
    resolution_pairs = resolution.split(',')
    parsed_resolutions = []
    for pair in resolution_pairs:
        width, height = map(int, pair.split('x'))
        parsed_resolutions.append([width, height])
    
    return parsed_resolutions
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Stable Diffusion ONNX models to RKNN models')
    parser.add_argument('-m','--model-dir', type=str, help='Directory containing the Stable Diffusion ONNX models', required=True)
    parser.add_argument('-c','--components', type=str, help='Name of the components to convert, e.g. "text_encoder,unet,vae_decoder"', default='text_encoder, unet, vae_decoder')
    parser.add_argument('-r','--resolutions', type=str, help='Comma-separated list of resolutions for the model, e.g. "256x256,512x512"', default='256x256')
    parser.add_argument('--target_platform', type=str, help='Target platform for the RKNN model, default is "rk3588"', default='rk3588')
    args = parser.parse_args()

    components = args.components.split(',')

    for component in components:
        onnx_path = f'{args.model_dir}/{component.strip()}/model.onnx'
        resolution_list = parse_resolution_list(args.resolutions)
        if(len(resolution_list) == 0):
            print("Error: No resolutions specified")
            exit(1)

        convert_pipeline_component(onnx_path, resolution_list, args.target_platform)
    

    
