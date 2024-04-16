import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': '/home/liulei/RGBT/pytracking/ltr/tensorboard/',
        'gtot_dir':'/home/Datasets/GTOT/',

        'gtot01_strict_dir':'/home/Datasets/GTOT_strict_01/',
        'gtot10_strict_dir':'/home/Datasets/GTOT_strict_10/',
        
        # 'gtot01_1_dir':'/data/Datasets/GTOT_1_01/',
        # 'gtot10_1_dir':'/data/Datasets/GTOT_1_10/',
        
        # 'gtot01_2_dir':'/data/Datasets/GTOT_2_01/',
        # 'gtot10_2_dir':'/data/Datasets/GTOT_2_10/',
        
        # 'gtot01_3_dir':'/data/Datasets/GTOT_3_01/',
        # 'gtot10_3_dir':'/data/Datasets/GTOT_3_10/',
        
        'gtot01_4_dir':'/home/Datasets/GTOT_40_01/',
        'gtot10_4_dir':'/home/Datasets/GTOT_40_10/',

        # 'gtot01_human_dir':'/data/Datasets/GTOT_01_human/',
        # 'gtot10_human_dir':'/data/Datasets/GTOT_10_human/',
        'rgbt210_dir':'/home/Datasets/RGBT210/',
        'rgbt234_dir':'/home/Datasets/RGBT234/',

        'rgbt21001_strict_dir':'/home/Datasets/RGBT210_strict_01/',
        'rgbt21010_strict_dir':'/home/Datasets/RGBT210_strict_10/',
        'rgbt23401_strict_dir':'/home/Datasets/RGBT234_strict_01/',
        'rgbt23410_strict_dir':'/home/Datasets/RGBT234_strict_10/',

        # 'rgbt21001_1_dir':'/data/Datasets/RGBT210_1_01/',
        # 'rgbt21010_1_dir':'/data/Datasets/RGBT210_1_10/',
        # 'rgbt23401_1_dir':'/data/Datasets/RGBT234_1_01/',
        # 'rgbt23410_1_dir':'/data/Datasets/RGBT234_1_10/',

        # 'rgbt21001_2_dir':'/data/Datasets/RGBT210_2_01/',
        # 'rgbt21010_2_dir':'/data/Datasets/RGBT210_2_10/',
        # 'rgbt23401_2_dir':'/data/Datasets/RGBT234_2_01/',
        # 'rgbt23410_2_dir':'/data/Datasets/RGBT234_2_10/',

        # 'rgbt21001_3_dir':'/data/Datasets/RGBT210_3_01/',
        # 'rgbt21010_3_dir':'/data/Datasets/RGBT210_3_10/',
        # 'rgbt23401_3_dir':'/data/Datasets/RGBT234_3_01/',
        # 'rgbt23410_3_dir':'/data/Datasets/RGBT234_3_10/',

        'rgbt21001_4_dir':'/home/Datasets/RGBT210_40_01/',
        'rgbt21010_4_dir':'/home/Datasets/RGBT210_40_10/',
        'rgbt23401_4_dir':'/home/Datasets/RGBT234_40_01/',
        'rgbt23410_4_dir':'/home/Datasets/RGBT234_40_10/',


        # 'rgbt23401_human_dir':'/data/Datasets/RGBT234_01_human/',
        # 'rgbt23410_human_dir':'/data/Datasets/RGBT234_10_human/',
        'lsotb_tir_dir':'/home/Datasets/LSOTB-TIR/',
        'LasHeR_dir':'/home/Datasets/LasHeR/',

        'LasHeR01_strict_train_dir':'/home/Datasets/LasHeR_train_strict_01/',
        'LasHeR10_strict_train_dir':'/home/Datasets/LasHeR_train_strict_10/',
        'LasHeR01_strict_test_dir':'/home/Datasets/LasHeR_test_strict_01/',
        'LasHeR10_strict_test_dir':'/home/Datasets/LasHeR_test_strict_10/',

        # 'LasHeR01_1_train_dir':'/data/Datasets/LasHeR_train_1_01/',
        # 'LasHeR10_1_train_dir':'/data/Datasets/LasHeR_train_1_10/',
        # 'LasHeR01_1_test_dir':'/data/Datasets/LasHeR_test_1_01/',
        # 'LasHeR10_1_test_dir':'/data/Datasets/LasHeR_test_1_10/',

        # 'LasHeR01_2_train_dir':'/data/Datasets/LasHeR_train_2_01/',
        # 'LasHeR10_2_train_dir':'/data/Datasets/LasHeR_train_2_10/',
        # 'LasHeR01_2_test_dir':'/data/Datasets/LasHeR_test_2_01/',
        # 'LasHeR10_2_test_dir':'/data/Datasets/LasHeR_test_2_10/',

        # 'LasHeR01_3_train_dir':'/data/Datasets/LasHeR_train_3_01/',
        # 'LasHeR10_3_train_dir':'/data/Datasets/LasHeR_train_3_10/',
        # 'LasHeR01_3_test_dir':'/data/Datasets/LasHeR_test_3_01/',
        # 'LasHeR10_3_test_dir':'/data/Datasets/LasHeR_test_3_10/',

        'LasHeR01_4_train_dir':'/home/Datasets/LasHeR_train_40_01/',
        'LasHeR10_4_train_dir':'/home/Datasets/LasHeR_train_40_10/',
        'LasHeR01_4_test_dir':'/home/Datasets/LasHeR_test_40_01/',
        'LasHeR10_4_test_dir':'/home/Datasets/LasHeR_test_40_10/',

        # 'LasHeR01_random_dir':'/data/Datasets/LasHeR_01_random/',
        # 'LasHeR10_random_dir':'/data/Datasets/LasHeR_10_random/',

        # 'gtot01_001_dir':'/data/Datasets/GTOT_001_01/',
        # 'gtot10_001_dir':'/data/Datasets/GTOT_001_10/',
        # 'rgbt21001_001_dir':'/data/Datasets/RGBT210_001_01/',
        # 'rgbt21010_001_dir':'/data/Datasets/RGBT210_001_10/',
        # 'rgbt23401_001_dir':'/data/Datasets/RGBT234_001_01/',
        # 'rgbt23410_001_dir':'/data/Datasets/RGBT234_001_10/',
        # 'LasHeR01_001_train_dir':'/data/Datasets/LasHeR_train_001_01/',
        # 'LasHeR10_001_train_dir':'/data/Datasets/LasHeR_train_001_10/',
        # 'LasHeR01_001_test_dir':'/data/Datasets/LasHeR_test_001_01/',
        # 'LasHeR10_001_test_dir':'/data/Datasets/LasHeR_test_001_10/',

        # 'gtot01_005_dir':'/data/Datasets/GTOT_005_01/',
        # 'gtot10_005_dir':'/data/Datasets/GTOT_005_10/',
        # 'rgbt21001_005_dir':'/data/Datasets/RGBT210_005_01/',
        # 'rgbt21010_005_dir':'/data/Datasets/RGBT210_005_10/',
        # 'rgbt23401_005_dir':'/data/Datasets/RGBT234_005_01/',
        # 'rgbt23410_005_dir':'/data/Datasets/RGBT234_005_10/',
        # 'LasHeR01_005_train_dir':'/data/Datasets/LasHeR_train_005_01/',
        # 'LasHeR10_005_train_dir':'/data/Datasets/LasHeR_train_005_10/',
        # 'LasHeR01_005_test_dir':'/data/Datasets/LasHeR_test_005_01/',
        # 'LasHeR10_005_test_dir':'/data/Datasets/LasHeR_test_005_10/',
        'gtot_sub_dir':'/home/Datasets/GTOT_sub/',
        'rgbt210_sub_dir':'/home/Datasets/RGBT210_sub/',
        'rgbt234_sub_dir':'/home/Datasets/RGBT234_sub/',
        'LasHeR_sub_dir':'/home/Datasets/LasHeR_sub/',
        })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'ltr.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))
