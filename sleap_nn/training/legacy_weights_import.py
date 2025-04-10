# import h5py

# h5_path = "sleap_nn/training/best_model.h5"
# with h5py.File(h5_path, "r") as f:
#     for layer_name in f['model_weights']:
#         print("LAYER_NAME:", layer_name)
#         for weight_name in f['model_weights'][layer_name]:
#             # print("WEIGHT_NAME:", weight_name)
#             print(f"{layer_name}/{weight_name} → shape: {f['model_weights'][layer_name][weight_name]}")

import h5py

h5_path = "sleap_nn/training/best_model.h5"
with h5py.File(h5_path, "r") as f:
    model_weights = f["model_weights"]
    
    for layer_name in model_weights:
        print(f"Layer: {layer_name}")
        layer_group = model_weights[layer_name]
        
        for weight_name in layer_group:
            weight_group = layer_group[weight_name]
            
            # Check if this is a dataset or another group
            if isinstance(weight_group, h5py.Dataset):
                print(f"  {layer_name}/{weight_name} → shape: {weight_group.shape}")
            elif isinstance(weight_group, h5py.Group):
                for param_name in weight_group:
                    param = weight_group[param_name]
                    if isinstance(param, h5py.Dataset):
                        print(f"  {layer_name}/{weight_name}/{param_name} → shape: {param.shape}")
