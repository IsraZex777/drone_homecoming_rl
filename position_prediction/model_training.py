from model import create_ann_model
from data_preprocessing import load_preprocessed_dataset


def train_model():
    train_x, train_y, dev_x, dev_y, scaler_x, scaler_y = load_preprocessed_dataset()

    model = create_ann_model()
    history = model.fit(train_x,
                        train_y,
                        epochs=250,
                        batch_size=512,
                        validation_data=(dev_x, dev_y))

    return model, scaler_x, scaler_y, history
