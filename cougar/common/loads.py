import hjson


def read_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        ret = f.read()
    return hjson.loads(ret)