# import the necessary packages
import os
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "E:\Training Data\Droplet Pictures and Labels\\2024 fr"
BOX_PATH = "E:\Training Data\Droplet Pictures and Labels\\2024 circlebox"
CIRCLEBOX_PATH = "E:\Training Data\Droplet Pictures and Labels\\2024 dual cb2c"
IMAGES_PATH = BASE_PATH # os.path.sep.join([BASE_PATH, "images"])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training plot,
# and testing image filenames
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "master.csv"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

C1_BOX_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector-c1box-2024-fullimg.keras"])
C1_BOX_ANNOTS_PATH = os.path.sep.join([BOX_PATH, "c1box.csv"])
C1_BOX_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "c1boxplot.png"])

C2_BOX_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector-c2box-2024-fullimg.keras"])
C2_BOX_ANNOTS_PATH = os.path.sep.join([BOX_PATH, "c2box.csv"])
C2_BOX_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "c1boxplot.png"])

DUAL_B2C_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector-dualb2c-2024.keras"])
DUAL_B2C_ANNOTS_PATH = os.path.sep.join([CIRCLEBOX_PATH, "cb2c.csv"])
DUAL_B2C_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "dualb2cplot.png"])

BB_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector-bb-2024 fr.keras"])
BB_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "bb.csv"])
BB_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "bbplot.png"])

TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 64