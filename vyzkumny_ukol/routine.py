import ModelEnsembling as ME
import LHmode_classifier as LH
import alt_models_training as amtr

from importlib import reload

if __name__ == "__main__":
    LH.train_and_test_ris_model(ris_option = 'RIS1',
                                num_workers = 32,
                                num_epochs_for_fc = 10,
                                num_epochs_for_all_layers = 10,
                                num_classes = 2,
                                batch_size = 32,
                                learning_rate_min = 0.001,
                                learning_rate_max = 0.01,
                                comment_for_model_name = f'Testing augmentation with RIS1',
                                random_seed = 42,
                                augmentation = True)
    
    LH.train_and_test_ris_model(ris_option = 'both',
                                num_workers = 32,
                                num_epochs_for_fc = 10,
                                num_epochs_for_all_layers = 10,
                                num_classes = 2,
                                batch_size = 32,
                                learning_rate_min = 0.001,
                                learning_rate_max = 0.01,
                                comment_for_model_name = f'Testing augmentation on both RIS1 and RIS2',
                                random_seed = 42,
                                augmentation = True)
    
    LH.train_and_test_ris_model(ris_option = 'both',
                                num_workers = 32,
                                num_epochs_for_fc = 10,
                                num_epochs_for_all_layers = 10,
                                num_classes = 2,
                                batch_size = 32,
                                learning_rate_min = 0.001,
                                learning_rate_max = 0.01,
                                comment_for_model_name = f' RIS1 and RIS2 without augmentation',
                                random_seed = 42,
                                augmentation = False)