# Neural Coarse-graining (NCG) reference implementation

This is an implementation of a 1D convolutional version of neural coarse-graining. It can be trained on a space-separated multi-channel signal to produce coarse-grained classes. The network architecture here uses two convolutional layers for the transformer and two convolutional layers for the predictor, all with batch normalization.

Command-line options:

* --input: File containing training data, which is also the data processed into the coarse-grained classes and predictions
* --valid: File containing validation data
* --output: File to write the coarse-grained classes of the input into after training
* --preds: File to write the predictor's predictions of the coarse-grained classes
* --error: File to write the training error curve into (if validation data is provided, also includes the validation error). The file contains columns for the total loss, as well as the separate average entropy and cross-entropy terms.
* --process: Just process input into predictions (with an existing trained model), rather than training
* --load_model: Filename of an existing trained model to load in
* --save_model: File to write the trained model into
* --epochs: Number of epochs to train for (defaults to 100)
* --lr: Learning rate (defaults to 1e-3)
* --dt: How far in the future to predict (defaults to 10 time steps)

Specifying the network architecture:

* --tr_filt1: Number of filters in the 1st transformer layer (default 20)
* --tr_filt2: Number of filters in the 2nd transformer layer (default 20)
* --pr_filt1: Number of filters in the 1st predictor layer (default 20)
* --pr_filt2: Number of filters in the 2nd predictor layer (default 20)
* --tr_fs1: Filter size of the 1st transformer layer (default 3)
* --tr_fs2: Filter size of the 2nd transformer layer (default 3)
* --pr_fs1: Filter size of the 1st predictor layer (default 3)
* --pr_fs2: Filter size of the 2nd predictor layer (default 3)
