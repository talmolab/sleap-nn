import json
from omegaconf import omegaconf


def convert_legacy_json_to_omegaconf(json_file_path, config_spec):
    '''
    this does some cool stuff here :)
    '''

    # 1. read legacy JSON file
    with open(json_file_path, 'r') as file:
        legacy_dict = json.load(file)

    # 2. map legacy dictionary fields to OmegaConf structure
    mapped_dict = {}
    for old_key, value in legacy_dict.items():
        new_key = get_new_key_from_old(old_key)
        mapped_dict[new_key] = value
        
    def get_new_key_from_old(old_key):
        mapping = {

        }
        return mapping.get(old_key, old_key)  # defaults to old_key if no mapping

    # 3. create OmegaConf dictionary
   config = OmegaConf.create(mapped_dict)
   return config