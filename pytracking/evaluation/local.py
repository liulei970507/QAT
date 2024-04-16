from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.gtot_path = '/home/Datasets/GTOT/'
    settings.lasher_path = '/home/Datasets/LasHeR/'
    settings.lashertestingSet_path = '/home/Datasets/LasHeR/'
    settings.network_path = '/home/liulei/RGBT/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.result_plot_path = '/home/liulei/RGBT/pytracking/pytracking/result_plots/'
    settings.results_path = '/home/liulei/RGBT/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.rgbt210_path = '/home/Datasets/RGBT210/'
    settings.rgbt234_path = '/home/Datasets/RGBT234/'
    settings.segmentation_path = '/home/liulei/RGBT/pytracking/pytracking/segmentation_results/'

    return settings

