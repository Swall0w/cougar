import torch
import argparse
from cougar.common import read_config, dump_config
import os
from cougar.common.comm import synchronize, get_rank
from cougar.common import collect_env_info
from pathlib import Path
import json
from cougar.common import setup_logger
from cougar.agents import build_agent


def main(config, args):
    agent_class = build_agent(config)
    agent = agent_class(config, args)
    agent.run()
    agent.finalize()


def arg():
    parser = argparse.ArgumentParser(description='PyTorch Model Trainer')
    parser.add_argument('-c', '--config', type=str,
                        default='configs/mnist.json',
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    config = read_config(args.config)

    # Experiment Setup
    output_dir = Path(config['experiment']['output_dir']) / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(config['experiment']['name'], output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config))
    logger.info("Running with config:\n{}".format(
        json.dumps(config, sort_keys=False, indent=4)
    ))

    # config save
    output_config_path = str(output_dir / 'config.json')
    logger.info("Saving config into: {}".format(output_config_path))
    dump_config(output_config_path, config)

    main(config, args)
