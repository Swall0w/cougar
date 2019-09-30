from cougar.agents import YOLOAgent
import argparse
from test_tube import Experiment
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint



def main():
    args = arg()
    exp = Experiment(save_dir='./result')

    agent = YOLOAgent(args)

    checkpoint_callback = ModelCheckpoint(
        filepath='./result',
        save_best_only=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(
        experiment=exp,
        max_nb_epochs=args.epochs,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
    )
    trainer.fit(agent)


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="configs/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="configs/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,
                        default='/home/fujitake/PyTorch-YOLOv3/weights/darknet53.conv.74',
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    main()
