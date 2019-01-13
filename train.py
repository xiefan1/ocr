import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from model import create_model

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

checkpoint_path = "checkpoints/cp-{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)
#tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)
model = create_model()

print('-----------Start training-----------')
#model.fit(x_train, y_train, epochs=5, callbacks = [cp_callback, tensorboard])
model.fit(x_train, y_train, epochs=5, callbacks = [cp_callback])
model.summary()
model.save('trained_model.h5')

#model.load_weights(checkpoint_path)
loss, acc = model.evaluate(x_test, y_test)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))

""" untrained model """
model = create_model()

loss, acc = model.evaluate(x_test, y_test)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

