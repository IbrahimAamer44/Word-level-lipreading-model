
from LRCN import create_LRCN_model

weights_path = "./Weights/Model_1-Without-CTC-LOSS_checkpoint1_tillbatch_9.h5"
model = create_LRCN_model()

model.load_weights(weights_path)

print("Model loaded Succesfully compiled !!!!")