# PPO implementation (Variant II)

## PPO

![ddpg](static/ppo_pseudocode.png)


## Analysis 

### Different sizes of NN

![](static/Figure_1.png)

### Model with mean and std

![](static/Figure_2.png)

### Things one can compare with other example implementations

Item | my run | [example 1](https://github.com/zzzxxxttt/pytorch_simple_RL/blob/master/ppo_mtcar.py) | [example 2](https://github.com/Abhipanda4/PPO-PyTorch) |
 --- | --- | --- | ---
| NN | --- | --- | --- |
 --- | --- | --- | ---
| batch size | --- | --- | --- |
| GAMMA | --- | --- | --- |
| LAMBDA | --- | --- | --- |
| ALPHA | --- | --- | --- |
 --- | --- | --- | ---


# !!! What was important:



## Credits:

### PPO

- [Environments in OpenAI (Leaderboard)](https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2)
- [PPO (OpenAI's blog)](https://openai.com/blog/openai-baselines-ppo/)
- [PPO implementation from Deep-Reinforcement-Learning-Hands-On-Second-Edition (page 606)](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter12/02_pong_a2c.py)
- [Optimization In Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#automatic-optimization)
- [Adam Grad - page 36 (Training NNs from Stanford's course)](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf)
- [Kullback–Leibler divergence (wikipedia)](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Kullback–Leibler divergence (YouTube video) - great](https://www.youtube.com/watch?v=ErfnhcEV1O8&ab_channel=Aur%C3%A9lienG%C3%A9ron)
- [1 - PPO implementation](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=yr-ZjT_CGyEi)
- [2 - PPO implementation with PL](https://github.com/sid-sundrani/ppo_lightning/blob/master/ppo_model.py)
- [3 - PPO implementation - OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)








