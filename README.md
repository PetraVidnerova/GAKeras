# GAKeras

Search for an optimal KERAS architecture using genetic algorithms. Uses DEAP for GA.

Tested on air pollution prediction task and MNIST. 

Edit config.py or use --config to set up the configuration. 


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
  --config              config file name
			
```
TRAINSET and TESTSET are either filenames or keywords ```mnist.train | mnist.test | mnist2d.train | mnist2d.test```.
Use either 1D data or main_conv.py for 2D data. 


## Examples

```
main.py --trainset mnist.train --testset mnist.test
main_es.py --trainset mnist.train --testset mnist.test 
main_conv.py --trainset mnist2d.train --testset mnist2d.test --config config_mnist.json
```

## Citation

**Vidnerová, Petra and Neruda, Roman**
*Evolving Keras Architectures for Sensor Data Analysis.* Proceedings of the 2017 Federated Conference
on Computer Science and Information Systems. Warszawa: Polish Information Processing Society, 2017 -
(Ganzha, M.; Maciaszek, L.; Paprzycki, M.), s. 109-112.
Annals of Computer Science and Information Systems, 11. ISBN 978-83-946253-7-5. ISSN 2300-5963.
[FedCSIS 2017. Federated Conference on Computer Science and Information Systems. Prague (CZ),
03.09.2017-06.09.2017] 

**Vidnerová, Petra and Neruda, Roman**
*Evolution Strategies for Deep Neural Network Models Design.*
Proceedings ITAT 2017: Information Technologies - Applications and Theory.
Aachen & Charleston: Technical University & CreateSpace Independent Publishing Platform,
2017 - (Hlaváčová, J.), s. 159-166. CEUR Workshop Proceedings, V-1885. ISBN 978-1974274741. ISSN 1613-0073.
[ITAT 2017. Conference on Theory and Practice of Information Technologies - Applications and Theory/17./.
Martinské hole (SK), 22.09.2017-26.09.2017]
[http://ceur-ws.org/Vol-1885/159.pdf](http://ceur-ws.org/Vol-1885/159.pdf) 