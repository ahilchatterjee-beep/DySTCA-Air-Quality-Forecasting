from Training.train import train
import config

def main():

    if config.MODE == "train":
        train()

    #elif config.MODE == "evaluate":
        #evaluate()

    #elif config.MODE == "predict":
        #predict()


if __name__ == "__main__":
    main()