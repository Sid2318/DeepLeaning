from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    
    for i in range(hp.Int('num_layers', min_value=1, max_value=10)):
        if i == 0:
            model.add(Dense(
                units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
                activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid']),
                input_dim=8
            ))
        else:
            model.add(Dense(
                units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
                activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid'])
            ))
        model.add(Dropout(
            rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.9, step=0.1)
        ))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile with tunable optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Adadelta(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
