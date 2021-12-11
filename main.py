from artificial_gps.experiment1 import train_model
from artificial_gps.data import load_preprocessed_sequences

if __name__ == "__main__":
    # output = load_preprocessed_sequences()
    # print(output[0][1000])
    # print(output[1][1000])
    output = train_model()