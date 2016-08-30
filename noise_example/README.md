Example using an alternating mix of unit Gaussian IID noise and a -1,1 Bernoulli process (also with unit standard deviation).
The noise types are interpolated according to a weight (1+cos(2*pi*t/500))/2.

To run this example:

python ../ncg.py --input train.txt --valid valid.txt --output cg.txt --preds preds.txt --error error.txt  --epochs 500

This will generate 3 coarse-grained variables (the script default), which after training will mark out different phases of the weight function.
