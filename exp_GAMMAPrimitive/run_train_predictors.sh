# python exp_GAMMAPrimitive/utils/utils_canonicalize_amass.py 1
# python exp_GAMMAPrimitive/utils/utils_canonicalize_amass.py 10
# python exp_GAMMAPrimitive/train_GammaPredictor.py --cfg Gamma_predictor_guggenheim_v0
# python exp_GAMMAPrimitive/train_GammaPredictor.py --cfg Gamma_predictor_guggenheim_v1 --resume_training 1
# cp results/exp_GAMMAPrimitive/Gamma_predictor_guggenheim_v0/checkpoints/epoch-300.ckp results/exp_GAMMAPrimitive/Gamma_predictor_guggenheim_v1/checkpoints/epoch-000.ckp
python exp_GAMMAPrimitive/train_GammaPredictor.py --cfg Gamma_predictor_guggenheim_v1 --resume_training 1
python exp_GAMMAPrimitive/train_GAMMAPolicyPPOToLocation_Guggenheim.py