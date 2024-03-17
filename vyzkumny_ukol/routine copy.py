import ModelEnsembling as ME
import LHmode_classifier as LH
import alt_models_training as amtr

from importlib import reload

if __name__ == "__main__":
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            batch_size = 128,
                            num_workers = 6,
                            num_epochs = 24,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils')
    

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 320,
                            batch_size = 128,
                            num_workers = 6,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils')
    
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 640,
                            batch_size = 128,
                            num_workers = 6,
                            num_epochs = 24,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils')
    

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 640,
                            batch_size = 128,
                            num_workers = 6,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, 4 mirnov_coils')
    
    amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                            architecture = 'Simple1DCNN',
                            signal_window = 320,
                            batch_size = 128,
                            num_workers = 6,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, h_alpha')

    amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            batch_size = 128,
                            num_workers = 6,
                            num_epochs = 12,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f' testing extrasensory model, h_alpha')