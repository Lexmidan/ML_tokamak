
import LHmode_classifier as LH
import alt_models_training as amtr

from importlib import reload

if __name__ == "__main__":
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 160,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 16,
                            dpoints_in_future = 80,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' exponential ELM decay, 4 mirnov_coils, 160 window, 150 sampling_freq')
    
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 160,
                            sampling_freq = 300,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 16,
                            dpoints_in_future = 80,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' exponential ELM decay, 4 mirnov_coils, 160 window, 300 sampling_freq')    
    
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 160,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 16,
                            dpoints_in_future = 80,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' exponential ELM decay, 4 mirnov_coils, 160 window, 150 sampling_freq')   

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 160,
                            sampling_freq = 300,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 16,
                            dpoints_in_future = 80,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' exponential ELM decay, 4 mirnov_coils, 160 window, 300 sampling_freq')         
    # LH.train_and_test_ris_model(ris_option = 'both',
    #                             num_workers = 32,
    #                             num_epochs_for_fc = 10,
    #                             num_epochs_for_all_layers = 10,
    #                             num_classes = 2,
    #                             batch_size = 32,
    #                             learning_rate_min = 0.001,
    #                             learning_rate_max = 0.01,
    #                             comment_for_model_name = f', without augmentation',
    #                             random_seed = 42,
    #                             augmentation = False)
    
    # LH.train_and_test_ris_model(ris_option = 'RIS1',
    #                             num_workers = 32,
    #                             num_epochs_for_fc = 10,
    #                             num_epochs_for_all_layers = 10,
    #                             num_classes = 2,
    #                             batch_size = 32,
    #                             learning_rate_min = 0.001,
    #                             learning_rate_max = 0.01,
    #                             comment_for_model_name = f', augmentation',
    #                             random_seed = 42,
    #                             augmentation = True)

    # LH.train_and_test_ris_model(ris_option = 'both',
    #                             num_workers = 32,
    #                             num_epochs_for_fc = 10,
    #                             num_epochs_for_all_layers = 10,
    #                             num_classes = 2,
    #                             batch_size = 32,
    #                             learning_rate_min = 0.001,
    #                             learning_rate_max = 0.01,
    #                             comment_for_model_name = f', augmentation',
    #                             random_seed = 42,
    #                             augmentation = True)