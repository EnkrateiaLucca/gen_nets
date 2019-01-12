

def compile_model(optimizer, plot = False):
    siam = Siamese1()
    x_train, y_train, x_test, y_test = siam.get_data_prep(name="fashion_mnist")
    tr_pairs, tr_y = siam.pairs(x_train,y_train)
    te_pairs, te_y = siam.pairs(x_test, y_test)
    input_shape=(x_train.shape[1:])
    siamese_model = siam.get_model(input_shape=input_shape, nnet=True)
    adam = siam.optimizers(name = optimizer)
    siamese_model.compile(optimizer=adam, loss=siam.c_loss_1, metrics = [siam.accuracy])
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(time()), write_graph=True)
    history = siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_split=0.2,
                  batch_size=128, callbacks = [tensorboard], epochs=10)

    if plot = True:
        siam.plot_training(history)

def get_accuracy(population):
    """
    recevies population returns accuracy averages
    """
    compile_model
