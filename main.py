from src.train import train_model
from src.config.args import get_train_args


def main():
    args = get_train_args()
    train_model(args)


if __name__ == "__main__":
    main()
