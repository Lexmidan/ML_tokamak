
import LHmode_classifier as LH
import alt_models_training as amtr
import Cross_validation as cval

import torchvision
print('befor import resnet weights')
from torchvision.models.resnet import ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights

from importlib import reload

if __name__ == "__main__":
    try:
        amtr.train_and_test_alt_model(signal_name = 'mc',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 160,
                                    learning_rate_max = 0.01,
                                    num_classes=2,
                                    weight_decay=1e-5,
                                    comment_for_model_name = f'no L-mode, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=False,
                                    use_ELMs=True,
                                    no_L_mode=True)
    except Exception as e:
        with open('./runs/logs/exception.log', 'w') as f:
            f.write(str(e) + '\n')

    try:
        amtr.train_and_test_alt_model(signal_name = 'mc',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 0,
                                    learning_rate_max = 0.01,
                                    num_classes=2,
                                    weight_decay=1e-5,
                                    comment_for_model_name = f'no L-mode, 0 dpoints_in_future, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=False,
                                    use_ELMs=True,
                                    no_L_mode=True)
    except Exception as e:
        with open('exception.log', 'w') as f:
            f.write(str(e) + '\n')

    try:
        amtr.train_and_test_alt_model(signal_name = 'mc',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 160,
                                    learning_rate_max = 0.01,
                                    num_classes=2,
                                    weight_decay=0.1,
                                    comment_for_model_name = f'no L-mode, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=False,
                                    use_ELMs=True,
                                    no_L_mode=True)
    except Exception as e:
        with open('exception.log', 'w') as f:
            f.write(str(e) + '\n')

    try:
        amtr.train_and_test_alt_model(signal_name = 'mc',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 0,
                                    learning_rate_max = 0.01,
                                    num_classes=2,
                                    weight_decay=0.1,
                                    comment_for_model_name = f'no L-mode, 0 dpoints_in_future, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=False,
                                    use_ELMs=True,
                                    no_L_mode=True)
    except Exception as e:
        with open('exception.log', 'w') as f:
            f.write(str(e) + '\n')

    # try:
    #     LH.train_and_test_ris_model(ris_option = 'both',
    #                                 pretrained_model=torchvision.models.resnet34(weights=ResNet101_Weights.IMAGENET1K_V1),
    #                                 num_workers = 32,
    #                                 num_epochs_for_fc = 10,
    #                                 num_epochs_for_all_layers = 10,
    #                                 num_classes = 3,
    #                                 batch_size = 32,
    #                                 learning_rate_min = 0.001,
    #                                 learning_rate_max = 0.01,
    #                                 comment_for_model_name = f', 3 classes, resnet101',
    #                                 random_seed = 42,
    #                                 augmentation = False)
    # except Exception as e:
    #     with open('exception.log', 'w') as f:
    #         f.write(str(e) + '\n')