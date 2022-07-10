from pip import main
from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron

def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)
    X, y = prepare_data(df_AND)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df_AND, model, plotName)


if __name__ == "__main__":

    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=AND, modelName="and.model",plotName="and.png", eta=ETA, epochs=EPOCHS)

