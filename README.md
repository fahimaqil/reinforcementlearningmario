# Deep Reinforcement Learning With Super Mario Bros

![alt text](/mario.jpg)


## Description
* Context: Super Mario Bros (SMB), a popular action game with a “real-life” environment and large state space is a perfect platform to design a Reinforcement Learning agent that can play a computer game. The agent requires to interact with various objects and obstacles in the world, encouraging a knowledge-rich approach to learning. 

* Method- We used the framework provided by OpenAI Gym. Super Mario Bros gym and extracted the information from the game environment to train an RL agent using PPO. We also introduced preprocessing methods such as frame scaling, stochastic frameskip, frame stacking, and noisy net to the environment to improve the performance of our agent. A variation of PPO is created by introducing a rollback operation to improve the stability of the training. 

* Results: The approach managed to train an agent that was able to complete the level after 20 hours of training. We successfully implemented a method that can perform better than the general PPO implementation with a 50% increase in performance without data preprocessing applied and 10 % with data preprocessing applied.

## Tools

* Pytorch has been used as the main machine learning library 
* We used OpenCV, a popular open-source computer vision library, to perform some of the preprocessing methods due to the SMB environment such as resizing the frames and turning the colour of the frames to RGB
* We used Python language for the implementation since it is the best language for machine learning, and it integrated well with both OpenCV2 and Pytorch. 
* Another python package such as numpy is also included in the implementation.
* The computational power of Google Colab was used in order to conduct the experiment.
* https://github.com/Kautenja/gym-super-mario-bros for the Mario gym environment.

## RL Agent Training Process and Level Completion

* Progress of training from zero knowledge to completing the level

[![Click to watch](https://img.youtube.com/vi/GsP5JRhEMQ0/0.jpg)](https://youtu.be/GsP5JRhEMQ0 "Click to watch")

## More Details of The Project

The full details including our result can be viewed here: https://github.com/fahimaqil/reinforcementlearningmario/blob/master/report/report.pdf
