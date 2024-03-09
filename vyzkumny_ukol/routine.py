import ModelEnsembling as ME
import LHmode_classifier as LH
import alt_models_training as amtr

from importlib import reload

if __name__ == "__main__":
    comment = f'batch_size_64, num_epochs_24, learning_rate_max_0.001, AdamW, fixed scheduler'
    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'InceptionTime',
                            signal_window = 160,
                            batch_size = 64,
                            num_workers = 32,
                            num_epochs = 24,
                            learning_rate_max = 0.001,
                            comment_for_model_name = comment)
    

    amtr.train_and_test_alt_model(signal_name = 'mc',
                            architecture = 'Simple1DCNN',
                            signal_window = 160,
                            batch_size = 64,
                            num_workers = 32,
                            num_epochs = 12,
                            learning_rate_max = 0.001,
                            comment_for_model_name = f'batch_size_64, num_epochs_12, learning_rate_max_0.001, AdamW, fixed scheduler')

    