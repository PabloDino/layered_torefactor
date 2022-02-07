from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

model1_in = Input(shape=(27, 27, 1))
model1_out = Dense(300, input_dim=40, activation='relu', name='layer_1')(model1_in)
model1 = Model(model1_in, model1_out)

model2_in = Input(shape=(27, 27, 1))
model2_out = Dense(300, input_dim=40, activation='relu', name='layer_2')(model2_in)
model2 = Model(model2_in, model2_out)


concatenated = concatenate([model1_out, model2_out])
out = Dense(1, activation='softmax', name='output_layer')(concatenated)

merged_model = Model([model1_in, model2_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', 
metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc',
save_best_only=True, verbose=2)
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

merged_model.fit([model1_in, model2_in], y=out, batch_size=384, epochs=200,
             verbose=1, validation_split=0.1, shuffle=True, 
callbacks=[early_stopping, checkpoint])
