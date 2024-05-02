### Reinforcement Learning Project for Grad Level Data Mining and Machine Learning

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
  

Results -> The optimal velocity was 10, the optimal track_width was 0.6 out of the tests we ran
