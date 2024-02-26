import ModelEnsembling as ME
import LHmode_classifier as LH

from importlib import reload

if __name__ == "__main__":
    comment = f', 2 output classes, random seed 123'
    model, ris1_model_path = LH.train_and_test_ris_model(ris_option='RIS1', 
                                                        comment_for_model_name=comment, random_seed=123)
    model, ris2_model_path = LH.train_and_test_ris_model(ris_option='RIS2', 
                                                        comment_for_model_name=comment, random_seed=123)
    reload(ME)
    ME.train_and_test_ensembled_model(one_ris_models_paths={'RIS1':ris1_model_path, 'RIS2':ris2_model_path}, 
                                      comment_for_model_name=comment, random_seed=123)
    