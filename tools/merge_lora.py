import sys
from safetensors.torch import load_file, save_file

# Function to merge LoRA weights into the Stable Diffusion model
def merge_lora_to_sd(model_weights, lora_weights):
    # Assuming lora_weights contain parameters to merge into the model_weights
    # Modify model_weights using lora_weights here
    # This is a placeholder for the actual merging logic
    # You may need to merge LoRA weights into the specific layers
    for key in lora_weights:
        if key in model_weights:
            model_weights[key] += lora_weights[key]  # Example of adding weights
        else:
            model_weights[key] = lora_weights[key]  # Adding new keys if necessary
    return model_weights

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python merge_model_lora_to_safetensors.py <model_path> <lora_path> <output_name>")
        sys.exit(1)

    model_path = sys.argv[1]  # Path to the Stable Diffusion model
    lora_path = sys.argv[2]   # Path to the LoRA model
    output_name = sys.argv[3] # Desired output filename

    print(f"Loading Stable Diffusion model from {model_path}...")
    # Load the Stable Diffusion model weights using safetensors
    model_weights = load_file(model_path)

    print(f"Loading LoRA model from {lora_path}...")
    # Load the LoRA model weights using safetensors
    lora_weights = load_file(lora_path)

    # Merge LoRA weights into the Stable Diffusion model
    print("Merging LoRA weights into the Stable Diffusion model...")
    merged_model = merge_lora_to_sd(model_weights, lora_weights)

    # Save the merged model as .safetensors
    print(f"Saving the merged model as {output_name}...")
    save_file(merged_model, output_name)

    print(f"Model successfully saved as {output_name}")

if __name__ == "__main__":
    main()
