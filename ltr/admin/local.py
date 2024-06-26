class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/liulei/RGBT/pytracking/ltr/tensorboard/'    # Directory for tensorboard files.
        self.gtot_dir = '/home/Datasets/GTOT/'
        self.gtot01_strict_dir = '/home/Datasets/GTOT_strict_01/'
        self.gtot10_strict_dir = '/home/Datasets/GTOT_strict_10/'
        self.gtot01_4_dir = '/home/Datasets/GTOT_40_01/'
        self.gtot10_4_dir = '/home/Datasets/GTOT_40_10/'
        self.rgbt210_dir = '/home/Datasets/RGBT210/'
        self.rgbt234_dir = '/home/Datasets/RGBT234/'
        self.rgbt21001_strict_dir = '/home/Datasets/RGBT210_strict_01/'
        self.rgbt21010_strict_dir = '/home/Datasets/RGBT210_strict_10/'
        self.rgbt23401_strict_dir = '/home/Datasets/RGBT234_strict_01/'
        self.rgbt23410_strict_dir = '/home/Datasets/RGBT234_strict_10/'
        self.rgbt21001_4_dir = '/home/Datasets/RGBT210_40_01/'
        self.rgbt21010_4_dir = '/home/Datasets/RGBT210_40_10/'
        self.rgbt23401_4_dir = '/home/Datasets/RGBT234_40_01/'
        self.rgbt23410_4_dir = '/home/Datasets/RGBT234_40_10/'
        self.lsotb_tir_dir = '/home/Datasets/LSOTB-TIR/'
        self.LasHeR_dir = '/home/Datasets/LasHeR/'
        self.LasHeR01_strict_train_dir = '/home/Datasets/LasHeR_train_strict_01/'
        self.LasHeR10_strict_train_dir = '/home/Datasets/LasHeR_train_strict_10/'
        self.LasHeR01_strict_test_dir = '/home/Datasets/LasHeR_test_strict_01/'
        self.LasHeR10_strict_test_dir = '/home/Datasets/LasHeR_test_strict_10/'
        self.LasHeR01_4_train_dir = '/home/Datasets/LasHeR_train_40_01/'
        self.LasHeR10_4_train_dir = '/home/Datasets/LasHeR_train_40_10/'
        self.LasHeR01_4_test_dir = '/home/Datasets/LasHeR_test_40_01/'
        self.LasHeR10_4_test_dir = '/home/Datasets/LasHeR_test_40_10/'

        self.gtot_sub_dir = '/home/Datasets/GTOT_sub/'
        self.rgbt210_sub_dir = '/home/Datasets/RGBT210_sub/'
        self.rgbt234_sub_dir = '/home/Datasets/RGBT234_sub/'
        self.LasHeR_sub_dir = '/home/Datasets/LasHeR_sub/'