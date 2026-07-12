# Reproducibility -------------------------------------------------------------

BASE_SEED = 42


# Dataset ---------------------------------------------------------------------

# Motor imagery classes: left hand, right hand, feet, and tongue
N_CLASSES = 4

# Number of EEG channels used; the three EOG channels are excluded
N_EEG_CHANNELS = 22


# Preprocessing ---------------------------------------------------------------

# Band-pass filter cutoff frequencies in Hz
L_FREQ = 8
H_FREQ = 30

# Epoch window in seconds relative to cue onset
EPOCH_TMIN = 0.5
EPOCH_TMAX = 4.0

# Small value added to the standard deviation to avoid division by zero
NORMALIZATION_EPS = 1e-8


# Cross-validation ------------------------------------------------------------

# Number of stratified folds used for training the ensemble
N_FOLDS = 5


# EEGNet architecture ---------------------------------------------------------

# Dropout probability used to reduce overfitting
EEGNET_DROPOUT_RATE = 0.5

# Length of the temporal convolution kernel in samples
EEGNET_KERNEL_LENGTH = 125

# Number of temporal convolution filters
EEGNET_F1 = 8

# Depth multiplier for the depthwise spatial convolution
EEGNET_D = 2

# Number of pointwise convolution filters
EEGNET_F2 = 16

# Type of dropout layer used by EEGNet
EEGNET_DROPOUT_TYPE = "Dropout"


# EEGNet training -------------------------------------------------------------

# Initial learning rate used by the Adam optimizer
EEGNET_LEARNING_RATE = 0.001

# Maximum number of training epochs
EEGNET_MAX_EPOCHS = 500

# Number of trials processed in each training batch
EEGNET_BATCH_SIZE = 16

# Number of epochs without validation-loss improvement before stopping
EEGNET_EARLY_STOPPING_PATIENCE = 50


# CSP -------------------------------------------------------------------------

# Number of spatial components extracted by CSP
CSP_N_COMPONENTS = 4

# Covariance regularization method used by CSP
CSP_REG = "ledoit_wolf"


# LDA -------------------------------------------------------------------------

# Solver used to train the Linear Discriminant Analysis classifier
LDA_SOLVER = "svd"