import keras
import test_engine
from keras.models import Sequential
from keras.layers import Dense
import environment
actor = Sequential()
actor.add(Dense(50, input_dim=environment.state_size, activation='relu',
                kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
actor.add(Dense(50, activation='relu',
                kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
actor.add(Dense(environment.action_size, activation='tanh',
                kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
actor.load_weights("./save_model/robot_actor.h5")
test_engine.test_never_ending(actor.predict) 