import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50

    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 64 , 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
    

    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True
    learning_rate = .02
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    
    model_previous = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_previous, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs) 

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!
    #Here you edit the new one!
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True
    learning_rate = .02
    neurons_per_layer = [64, 64 , 10]

    model_improved = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    plt.figure(figsize=(14,8))
    plt.subplot(1, 2, 1)
    #utils.plot_loss(train_history["loss"],
    #                "Task 3 Model - Previous - Training loss", npoints_to_average=10)
    utils.plot_loss(    
        train_history_no_shuffle["loss"], "Task 4d Model - 60 hidden units * 2 - Training loss ", npoints_to_average=10)
    utils.plot_loss(    
        val_history_no_shuffle["loss"], "Task 4d Model - 60 hidden units * 2 - Validation loss ", npoints_to_average=10)
    utils.plot_loss(    
        train_history["loss"], "Task 4e Model - 64 hidden units * 10 - Training loss ", npoints_to_average=10)
    utils.plot_loss(    
        val_history["loss"], "Task 4e Model - 64 hidden units * 10 - Validation loss ", npoints_to_average=10)
    plt.ylim([0, .6])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    #utils.plot_loss(val_history["accuracy"], "Task 3 Model - Previous - Validation Accuracy")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 4d Model - 60 hidden units * 2 - Validation Accuracy")
    utils.plot_loss(
        train_history_no_shuffle["accuracy"], "Task 4d Model - 60 hidden units * 2 - Traning Accuracy")
    utils.plot_loss(
        val_history["accuracy"], "Task 4e Model - 64 hidden units * 10 - Validation Accuracy")
    utils.plot_loss(
        train_history["accuracy"], "Task 4e Model - 64 hidden units * 10 - Traning Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #CHANGE THIS
    plt.savefig("task4e_64_10.png")
    plt.show()
    
