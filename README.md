### Final Project for Data Mining and Machine Learning
Reinforcement Learning methods for solving a simulation of the inverted pendulum problem.


Main Findings: 
2a. Deep learning is not needed to balance the pendulum effectively, but it demonstrated improved performance.

2b. As seen in the evaluation charts, there is significant improvement with the use of the DQN. 
    Occasionally the Q-Learning algorithm yielded better results than the DQN, but these results were inconsistent.
    DQN demonstrated consistent results that tended to be better than traditional Q-learning. 

2c. 
  1Q. The parameters had drastic effects on the system, which can be seen in some of the attached plots.
      Velocity and track width were particularly influential on the system's performance. 
  2Q. We applied 5, 10, and 15 Newtons
      We paired each of these with 0.6, 1.2, and 2.4 meters in one direction (double to get full track length)
   
      *** Optimal Performance was reached with 10 newtons and 0.6 meters ***

      The average Reward Per Episode ranged between 100-120 with these parameters 
  
The higher the velocity and wider the track, the more the reward per episode seems to stabilize - at least for DQN

Optimal Results -> The optimal velocity was 10, and the optimal track_width was 0.6 out of the tests we ran


Videos from our experiments:

### Q-Learning
[![Q-Learning](https://img.youtube.com/vi/hhrLYEw1LoE/0.jpg)](https://youtu.be/hhrLYEw1LoE)

### Deep Q-Networks
[![Deep Q-Networks](https://img.youtube.com/vi/fBLAi6L_moY/0.jpg)](https://youtu.be/fBLAi6L_moY)

    ***Variables for Interpreting the Performance Measures***
    
    CR_Training = Cumulative Reward during training
    EL_Training = Episode Lengths during training
    CR_Evaluation = Cumulative Reward during model evaluation (after training)
    vel = velocity of cart
    track = width of the track in one direction from the center



