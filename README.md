### Reinforcement Learning Project for Grad Level Data Mining and Machine Learning

We created a simulation of the various principles. These are videos from our experiments.

### Q-Learning
[![Q-Learning](https://img.youtube.com/vi/hhrLYEw1LoE/0.jpg)](https://youtu.be/hhrLYEw1LoE)

### Deep Q-Networks
[![Deep Q-Networks](https://img.youtube.com/vi/fBLAi6L_moY/0.jpg)](https://youtu.be/fBLAi6L_moY)


When you're looking at the charts:

    ***These are good metrics to be able to extract performance measures.***
    
    CR_Training = Cumulative Reward during training
    EL_Training = Episode Lengths during training
    CR_Evaluation = Cumulative Reward during model evaluation (after training)
    vel = velocity of cart
    track = width of track in one direction from center




2a. You can do it with just Q_Learning, but Deep Learning definitely improves performance

2b. There is a lot of improvement with DRL (DQN), you can see from the evaluation charts. 
    Sometimes Q-Learning will yield better results, but they are much more inconsistent.
    DQN has much more consistent results and often better.

2c. 
  1Q. The parameters have drastic effects on the system. You can see from the various charts.
      Velocity and track_width especially
  2Q. We applied 5, 10, and 15 Newtons
      We paired each of these with 0.6, 1.2 and 2.4 meters in one direction (so double them to get full track length)
      But we're interested in the distance it moves from the center.

      *** Optimal Performance was reached with 10 newtons and 0.6 meters ***

      Average Reward Per Episode ranged between 100-120 with these parameters set
  

The higher the velocity and wider the track, the more the reward per episode seems to stabilize - at least for DQN

Results -> The optimal velocity was 10, the optimal track_width was 0.6 out of the tests we ran



