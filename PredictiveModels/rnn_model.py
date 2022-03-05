
from __future__ import absolute_import, division, print_function

# Import TensorFlow v2.
import tensorflow as tf
from tensorflow.keras import Model, layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#Neural network
class nn_model(Model):
    def __init__(
        self,
        size,
        output_size,
        num_layers = 1,
        masking_val = -1,
        size_layer = 50,
    ):
        super(nn_model, self).__init__()
        # Define a Masking Layer with -1 as mask.
        self.masking = layers.Masking(mask_value=masking_val)
        # Define a LSTM layer to be applied over the Masking layer.
        # Dynamic computation will automatically be performed to ignore -1 values.
        #self.lstm = layers.LSTM(units=num_units)

        rnn_cells = [layers.LSTMCell(size_layer) for _ in range(num_layers)]
        stacked_lstm = layers.StackedRNNCells(rnn_cells)
        self.lstm = layers.RNN(stacked_lstm)
        self.logits = layers.Dense(units = 1)
    
    def call(self, x, timestamp, is_training = False):
        # A RNN Layer expects a 3-dim input (batch_size, seq_len, num_features).
        x = tf.reshape(x, shape=[-1, timestamp, 1])
        
        # Apply Masking layer.
        x = self.masking(x)
        
        # Apply LSTM layer.
        x = self.lstm(x)
        
        # Apply output layer.
        x = self.logits(x)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def main():

    #Reads csv file to forecast
    series = pd.read_csv('PredictiveModels\\Data\\MSFT.csv', header=0, index_col=0)
    #test period:
    test_size = 20
    epoch = 100
    learning_rate = 0.01
    timestamp = 60
    display_step = 100
    #Scaling data
    series_scaled = series.iloc[:,4:5].values
    sc = MinMaxScaler(feature_range = (0, 1))
    series_scaled = sc.fit_transform(series_scaled)
    #Divide data
    df_train = series_scaled[:-test_size]
    df_test = series_scaled[-test_size:]

    #Transforms data: LSTMs expect our data to be in a specific format (3D array)
    X_train = []
    y_train = []

    for i in range(60, len(df_train)-1):
        X_train.append(df_train[i-60:i, 0])
        y_train.append(df_train[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Use tf.data API to shuffle and batch data.
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    batch_size = 50
    #train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    train_data = train_data.repeat().batch(batch_size).prefetch(1)

    # Build LSTM model.
    lstm_net = nn_model(size = df_train.shape[1], output_size = df_train.shape[1])
    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate)

    # Cross-Entropy Loss.
    # Note that this will apply 'softmax' to the logits.
    def cross_entropy_loss(x, y):
        # Convert labels to int 64 for tf cross-entropy function.
 
        y = tf.cast(y, tf.int64)

        #y = tf.keras.Input(shape=[None], dtype = tf.int64)
        # Apply softmax to logits and compute cross-entropy.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        # Average loss across the batch.
        return tf.reduce_mean(loss)

    # Optimization process. 
    def run_optimization(x, y):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = lstm_net(x, timestamp, is_training=True)
            # Compute loss.
            loss = cross_entropy_loss(pred, y)
            
        # Variables to update, i.e. trainable variables.
        trainable_variables = lstm_net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)
        
        # Update weights following gradients.
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    # Run training for the given number of steps.
    for step, (batch_x, batch_y) in enumerate(train_data.take(epoch), 1):
        # Run the optimization to update W and b values.
        run_optimization(batch_x, batch_y)
        
        if step % display_step == 0 or step == 1:
            pred = lstm_net(batch_x, timestamp, is_training=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
    
    predicted_stock_price = sc.inverse_transform(pred)
    #real_price = sc.inverse_transform(batch_y)
    print(predicted_stock_price)        


if __name__ == '__main__':
    main()