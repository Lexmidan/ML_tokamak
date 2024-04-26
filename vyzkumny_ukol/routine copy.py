import ModelEnsembling as ME
import LHmode_classifier as LH
import alt_models_training as amtr


if __name__ == "__main__":
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 24,
                            learning_rate_max = 0.01,
                            dpoints_in_future = 320,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 320 window, 150 sampling_freq')

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 320,
                            sampling_freq = 150,
                            batch_size = 512,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 320 window, 150 sampling_freq')
    
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 640,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 24,
                            learning_rate_max = 0.01,
                            dpoints_in_future = 320,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 640 window, 150 sampling_freq')
    

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 640,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            dpoints_in_future = 320,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 640 window, 150 sampling_freq')
    
    amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                            architecture = 'Simple1DCNN',
                            signal_window = 320,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, h_alpha, 320 window, 150 sampling_freq')

    amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            sampling_freq = 150,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, h_alpha, 320 window, 150 sampling_freq')
    
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            sampling_freq = 30,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 24,
                            learning_rate_max = 0.01,
                            dpoints_in_future = 320,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 320 window, 30 sampling_freq')

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 320,
                            sampling_freq = 30,
                            batch_size = 512,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 320 window, 30 sampling_freq')
    
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 640,
                            sampling_freq = 30,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 24,
                            learning_rate_max = 0.01,
                            dpoints_in_future = 320,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 640 window, 30 sampling_freq')
    

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 640,
                            sampling_freq = 30,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            dpoints_in_future = 320,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils, 640 window, 30 sampling_freq')
    
    amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                            architecture = 'Simple1DCNN',
                            signal_window = 320,
                            sampling_freq = 30,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, h_alpha, 320 window, 30 sampling_freq')

    amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            sampling_freq = 30,
                            batch_size = 256,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, h_alpha, 320 window, 30 sampling_freq')