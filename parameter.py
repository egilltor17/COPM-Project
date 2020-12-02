import os
import sys
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')
# -----------------------Path related parameters---------------------------------------

if sys.platform == 'win32':
    base_path = os.path.abspath(os.getcwd()).replace('\\', '/') + '/'

    train_ct_path     = base_path + 'data/LITS/Training_Batch_2/'
    train_seg_path    = base_path + 'data/LITS/Training_Batch_2/'
    test_ct_path      = base_path + 'data/LITS/Training_Batch_1/'
    test_seg_path     = base_path + 'data/LITS/Training_Batch_1/'
    training_set_path = base_path + 'data/LITS/Testing_Batch_1/'
    pred_path         = base_path + 'data/LITS/prediction/' 
    crf_path          = base_path + 'data/LITS/crf/'
    module_path       = base_path + 'data/models/net.pth' 
else:
    base_path = '/content/drive/MyDrive/Colab Notebooks/COMP-Project/'
    train_ct_path     = base_path + 'train/Training Batch 2/'
    train_seg_path    = base_path + 'train/Training Batch 2/'
    test_ct_path      = base_path + 'train/Training Batch 1/'
    test_seg_path     = base_path + 'train/Training Batch 1/'
    training_set_path = base_path + 'test/'
    pred_path         = base_path + 'prediction/' 
    crf_path          = base_path + 'crf/'
    module_path       = base_path + 'models/net.pth'

# -----------------------Path related parameters---------------------------------------


# ---------------------Training data to obtain relevant parameters-----------------------------------

size = 48                                   # Use 48 consecutive slices as input to the network
down_scale = 0.5                            # Cross-sectional downsampling factor
expand_slice = 20                           # Only use the liver and the upper and lower 20 slices of the liver as training samples
slice_thickness = 1                         # Normalize the spacing of all data on the z-axis to 1mm
upper, lower = 200, -200                    # CT data gray cut window

# ---------------------Training data to obtain relevant parameters-----------------------------------


# -----------------------Network structure related parameters------------------------------------

drop_rate = 0.3

# -----------------------Network structure related parameters------------------------------------


# ---------------------Network training related parameters--------------------------------------

gpu = '0'
Epoch = 1000
learning_rate = 1e-4
learning_rate_decay = [500, 750]
alpha = 0.33                                # In-depth supervision attenuation coefficient
batch_size = 1
num_workers = 3
pin_memory = True
cudnn_benchmark = True

# ---------------------Network training related parameters--------------------------------------


# ----------------------Model test related parameters-------------------------------------

threshold = 0.5 
stride = 12
maximum_hole = 5e4

# ----------------------Model test related parameters-------------------------------------


# ---------------------CRF post-processing optimization related parameters----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30   # The number of expansions in three directions based on the predicted results
max_iter = 20                               # CRF iterations
s1, s2, s3 = 1, 10, 10                      # CRF Gaussian kernel parameters

# ---------------------CRF post-processing optimization related parameters----------------------------------