import pandas as pd
import tensorflow as tf
import numpy as np


def create_datasets(x_train_tens, y_train_tens, x_val_tens, y_val_tens, x_test_tens, y_test_tens,
                    buffer_size, batch_size):
    """
    Batches and prefetches the train, validation, and test sets to optimize performance.

    Args:
        x_train_tens (tf.Tensor): Training features tensor.
        y_train_tens (tf.Tensor): Training labels tensor.
        x_val_tens (tf.Tensor): Validation features tensor.
        y_val_tens (tf.Tensor): Validation labels tensor.
        x_test_tens (tf.Tensor): Test features tensor.
        y_test_tens (tf.Tensor): Test labels tensor.
        buffer_size (int): Buffer size for shuffling the dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing:
         - Batched and prefetched training dataset.
         - Batched and prefetched validation dataset.
         - Batched and prefetched testing dataset.
    """
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train_tens, y_train_tens)).shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (x_val_tens, y_val_tens)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test_tens, y_test_tens)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset


def define_optimizer(optimizer, learn_rate, mom):
    """
        Initializes the optimizer based on the specified type and parameters.

        Args:
            optimizer (str): The type of optimizer ('adam', 'sgd', etc.).
            learn_rate (float): The learning rate for the optimizer.
            mom (float): The momentum parameter for the SGD optimizer (ignored if using other
            optimizers).

        Returns:
            tf.keras.optimizers.Optimizer: The initialized optimizer.
        """
    # Initialize the optimizer based on the string input:
    if optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=mom)
    elif optimizer.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learn_rate)
    elif optimizer.lower() == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learn_rate)
    elif optimizer.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learn_rate)
    elif optimizer.lower() == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learn_rate)
    elif optimizer.lower() == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=learn_rate)
    else:
        raise ValueError("Unsupported optimizer. Please use 'adam', 'sgd', 'rmsprop', 'adagrad',"
                         "'adadelta', 'nadam' or 'ftrl'.")
    return optimizer


def create_model_train_eval_present_results(optimizer, learn_rate, mom, n_range, input_size,
                                            hidden_layer_sizes, activation_fun, output_size,
                                            activation_fun_output, loss_fun, patience, train_set,
                                            epochs, validation_set, test_set, verb, batch_size):
    """
    Trains, evaluates, and presents results of a neural network model by running multiple
    training and evaluation cycles. Returns a DataFrame containing the average and standard
    deviation of test accuracy and loss.

    Parameters:
    - optimizer (str): The optimization technique to use. Possible choices include:
        'adam': Adaptive Moment Estimation (Adam) optimizer.
        'sgd': Stochastic Gradient Descent (SGD) optimizer.
        'rmsprop': Root Mean Square Propagation (RMSprop) optimizer.
        'adagrad': Adaptive Gradient Algorithm (Adagrad) optimizer.
        'adadelta': Adaptive Delta (Adadelta) optimizer.
        'nadam': Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimizer.
        'ftrl': Follow-the-regularized-leader (FTRL) optimizer.
    - learn_rate (float): The learning rate for the optimizer.
    - mom (float or None): The momentum parameter for the SGD optimizer (ignored for other
    optimizers).
    - n_range (int): The number of times to run the training and evaluation cycles.
    - input_size (tuple): Shape of the input features (e.g., (number of features,)).
    - hidden_layer_sizes (list of int): List of integers where each integer represents the number
      of neurons in a hidden layer.
    - activation_fun (str): Activation function for the hidden layers. Possible choices include:
        'relu': Rectified Linear Unit activation.
        'sigmoid': Sigmoid activation function.
        'tanh': Hyperbolic tangent activation function.
        'linear': Linear activation (no activation).
        'softmax': Softmax activation (typically used in the output layer for classification).
    - output_size (int): Number of output units (typically number of classes).
    - activation_fun_output (str): Activation function for the output layer. Possible choices
    include:
        'relu': Rectified Linear Unit activation.
        'sigmoid': Sigmoid activation function.
        'tanh': Hyperbolic tangent activation function.
        'linear': Linear activation (no activation).
        'softmax': Softmax activation (used for multi-class classification).
    - loss_fun (str): Loss function to optimize. Possible choices include:
        'mean_squared_error': Mean Squared Error (MSE) loss.
        'mean_absolute_error': Mean Absolute Error (MAE) loss.
        'categorical_crossentropy': Categorical Cross-Entropy loss (for multi-class classification).
        'sparse_categorical_crossentropy': Sparse Categorical Cross-Entropy loss (for multi-class
        classification with sparse labels).
        'binary_crossentropy': Binary Cross-Entropy loss (for binary classification).
    - train_set (tf.data.Dataset): The training dataset.
    - epochs (int): Number of epochs to train the model.
    - validation_set (tf.data.Dataset): The validation dataset.
    - test_set (tf.data.Dataset): The test dataset.
    - verb (int): Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).

    Returns:
    - pd.DataFrame: A DataFrame containing the average and standard deviation of test accuracy
      and loss.
    """
    batch_size_here = batch_size
    model_accuracy = []
    model_loss = []

    # Run the training and evaluation multiple times to ensure robustness:
    for i in range(n_range):
        # Initialize the optimizer object in each run:
        optimizer_here = define_optimizer(optimizer, learn_rate, mom)
        # Initialize a Sequential model:
        model = tf.keras.Sequential()
        # Add the input layer with the specified input shape:
        model.add(tf.keras.layers.Input(shape=input_size))
        # Add hidden layers based on the provided sizes and activation function:
        for size in hidden_layer_sizes:
            model.add(tf.keras.layers.Dense(size, activation=activation_fun))
        # Add the output layer with the specified activation function:
        model.add(tf.keras.layers.Dense(output_size, activation=activation_fun_output))
        # Compile the model with the specified optimizer and loss function:
        model.compile(
            optimizer=optimizer_here,  
            loss=loss_fun,
            metrics=['accuracy']
        )
        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=patience,
            # restore_best_weights=True  # Restore the weights of the best epoch
        )
        # Train the model with the training set and validate on the validation set:
        model.fit(
            train_set,
            epochs=epochs,
            validation_data=validation_set,
            batch_size=batch_size_here,
            verbose=verb,
            callbacks=[early_stopping]
        )
        # Evaluate the model on the test set to assess performance:
        test_loss, test_accuracy = model.evaluate(test_set, verbose=verb)
        model_loss.append(test_loss)  # Add loss of each run to a list
        model_accuracy.append(test_accuracy)  # Add accuracy of each run to a list

    # Calculate average and standard deviation for accuracy and loss:
    average_test_acc = round(np.mean(model_accuracy), 4)
    std_test_acc = round(np.std(model_accuracy), 4)
    average_test_loss = round(np.mean(model_loss), 4)
    std_test_loss = round(np.std(model_loss), 4)

    # Create a dictionary to store the results:
    dict_final = {
        'Batch Size': batch_size_here,
        'Number of Runs': n_range,
        'Optimization Technique': optimizer,
        'Loss Function': loss_fun,
        'Learning Rate': learn_rate,
        'Momentum': mom,
        'Patience': patience,
        'Hidden Layers Act. Function': activation_fun,
        'Output Activation Function': activation_fun_output,
        'Epochs': epochs,
        'List of Hidden Layers': hidden_layer_sizes,
        'Test Accuracy Average': average_test_acc,
        'Test Accuracy Standard Deviation': std_test_acc,
        'Test Loss Average': average_test_loss,
        'Test Loss Standard Deviation': std_test_loss
    }
    # Convert the dictionary to a DataFrame for better readability:
    dict_final = pd.DataFrame([dict_final]).transpose()
    dict_final.columns = ['Value']
    return dict_final
