# models/

Trained model artefacts are written here by `federated_learning.training.save_model` as `models/<name>/<version>/{weights.npz, model_card.json}`.
The card records the configuration, metrics and the SHA-256 of the weights; `load_model` refuses weights that do not match it. Contents are git-ignored.
