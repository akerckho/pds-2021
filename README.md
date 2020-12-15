PACKAGE TO INSTALL :

car_racing.py:
	- numpy
	- pyglet
	- gym
	- Box2D
	
dqnCustom.py:
	- torch
	- torchvision

record_dataset.py
	- imageio

train.py
	- carbontracker
	- sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj



# Entrainement d'une IA
python3 record_dataset.py train_set # jouer un petit temps pour donner des informations en entrée du modèle
python3 record_dataset.py test_set  

python3 train.py

python3 drive.py $PWD/models2/model-5.weights

