# GAKeras

Search for an optimal KERAS architecture using genetic algorithms. Uses DEAP for GA.

## Usage

```
main.py [-h] [--trainset TRAINSET] [--testset TESTSET] [--id ID]
               [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --trainset TRAINSET   filename of training set
  --testset TESTSET     filename of test set
  --id ID               computation id
  --checkpoint CHECKPOINT
                        checkpoint file to load the initial state from
```