import json
from collections import OrderedDict


# config save
def dump_config(config_name: str, config: OrderedDict):
    with open(config_name, 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)