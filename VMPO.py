import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


  
def get_KL(prob1, logprob1, logprob2):
    kl = prob1 * (logprob1 - logprob2)
    return kl.sum(1, keepdim=True)
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_probs = dist.probs
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_probs
        
class VMPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True)
        self.eps_eta = 0.02
        self.eps_alpha = 0.1
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        
        params = [
                {'params': self.policy.parameters()},
                {'params': self.eta},
                {'params': self.alpha}
            ]
        
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
  
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        #old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # get old probs and old advantages
        with torch.no_grad():
            _, old_state_values , old_dist_probs = self.policy_old.evaluate(old_states, old_actions)
            advantages = rewards - old_state_values.detach()
            
        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating sampled actions and values :
            logprobs, state_values, dist_probs = self.policy.evaluate(old_states, old_actions)
        
            # Get samples with top half advantages
            advprobs = torch.stack((advantages,logprobs))
            advprobs = advprobs[:,torch.sort(advprobs[0],descending=True).indices]
            good_advantages = advprobs[0,:len(old_states)//2]
            good_logprobs = advprobs[1,:len(old_states)//2]
            
            # Get losses
            phis = torch.exp(good_advantages/self.eta.detach())/torch.sum(torch.exp(good_advantages/self.eta.detach()))
            L_pi = -phis*good_logprobs
            L_eta = self.eta*self.eps_eta+self.eta*torch.log(torch.mean(torch.exp(good_advantages/self.eta)))
            
            KL = get_KL(old_dist_probs.detach(),torch.log(old_dist_probs).detach(),torch.log(dist_probs))
         
            
            L_alpha = torch.mean(self.alpha*(self.eps_alpha-KL.detach())+self.alpha.detach()*KL)
        
            
            
            
            loss = L_pi + L_eta + L_alpha + 0.5*self.MseLoss(state_values, rewards) 

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            with torch.no_grad():
                self.eta.copy_(torch.clamp(self.eta,min=1e-8))
                self.alpha.copy_(torch.clamp(self.alpha,min=1e-8))
            if i == self.K_epochs-1:
                print(torch.mean(KL).item(),self.alpha.item())
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2400      # update policy every n timesteps
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 8            # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = VMPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render or running_reward > (log_interval*solved_reward)*0.8:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
