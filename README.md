# PRINTEMPS DES SCIENCES 2021 : GreenML
### PACKAGE TO INSTALL :

- car_racing.py:
  - numpy
  - pyglet
  - gym
  - Box2D
	
- dqnCustom.py:
  - torch
  - torchvision

- record_dataset.py
  - imageio

- train.py
  - carbontracker
  - Don't forget to : sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj


### ENTRAINEMENT D'UNE IA
* python3 record_dataset.py train_set 
* python3 record_dataset.py test_set  

* python3 train.py

* python3 drive.py $PWD/models2/model-5.weights

