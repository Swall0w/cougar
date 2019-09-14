import hjson


def read_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        ret = f.read()
    return hjson.loads(ret)


def read_labels(filename: str) -> list:
    with open(filename) as f:
        lines = f.read().splitlines()

    return lines
