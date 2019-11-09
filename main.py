import model.ResNet as Res
from trainer import Trainer


def main():

    res5 = Res.resnet5()

    model_trainer = Trainer(res101)

    if res101 is None:
        model_trainer.set_model(res101)
        model_trainer.train()


if __name__ == '__main__':
    main()
