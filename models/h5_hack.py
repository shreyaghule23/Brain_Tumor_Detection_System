import h5py

# Open the file in read/write mode
with h5py.File("models/vgg16_best.h5", "r+") as f:
    model_config = f.attrs.get("model_config")
    
    if model_config:
        # Handle both byte and string formats
        if hasattr(model_config, 'decode'):
            new_config = model_config.decode("utf-8")
        else:
            new_config = model_config

        # Clean up the problematic keys
        new_config = new_config.replace('"batch_shape"', '"batch_input_shape"')
        new_config = new_config.replace(', "quantization_config": null', '')
        
        # Save it back (h5py handles the conversion back to appropriate format)
        f.attrs.modify("model_config", new_config)
        print(" Model file fixed! Now try running your load_model script.")
    else:
        print("Could not find model_config in the file.")
