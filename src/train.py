from src.trainer.trainer import ImageNetTrainer


def train_model(args):
    trainer = ImageNetTrainer(args)
    trainer.train()
