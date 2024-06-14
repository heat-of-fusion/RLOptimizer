from IntegratedEnvironment import IntegratedEnvironment

import pickle as pkl
from tqdm.auto import tqdm

from A2CNets import PolicyNet, CriticNet
from DLModel import CNNet
from OptimzerAgent import OptimAgentDemoV2

from utils import *

prev_3rd_train_loss = list()
prev_2nd_train_loss = list()
prev_1st_train_loss = list()

prev_3rd_valid_loss = list()
prev_2nd_valid_loss = list()
prev_1st_valid_loss = list()

BENCHMARK = [
    'Adam',
    'SGD',
    'Adagrad',
    'RMSprop'
]

episode_dict_keys = [
    'probs_1', 'probs_2', 'probs_3',
    'log_probs_1', 'log_probs_2', 'log_probs_3',
    'values', 'rewards',
    'beta_1_list', 'beta_2_list', 'gamma_3_list',
    'train_loss_dict', 'valid_loss_dict',
    'train_accuracy_dict', 'valid_accuracy_dict'
]

MAX_EPISODES = 1000
MAX_EPOCHS = 100
DL_LR = 1e-3
RL_LR = 3e-4

TRAIN_LEN = 1000
VALID_LEN = 1000

LOSS_RANGE_X = (-3, MAX_EPOCHS)
LOSS_RANGE_Y = (1.45, 2.15)

REWARD_RANGE_X = (-3, MAX_EPISODES)
REWARD_RANGE_Y = (16.0, 32.0)

REWARD_ALPHA = 1.0

DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95

INPUT_SIZE = 1
HIDDEN_SIZE = 64
N_CLASSES = 10

batch_shape = [32, 1, 28, 28]

ACTION_DIM = 10

benchmark_episode = [*[value for value in range(0, 1000, 100)], 999]

PLOT_PATH = f'./history/plots/mnist/'
MODEL_PATH = f'./history/model/mnist/'
INFO_PATH = f'./history/episode_dict/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_dataset, valid_dataset = get_mnist()
random_sampler = RandomSampleLoader()
print(random_sampler.download_data())
train_loader, valid_loader, retrain_loader, revalid_loader = random_sampler.get_dataloader(TRAIN_LEN, VALID_LEN, first = True)

pkl.dump(retrain_loader, open(f'./retrain_loader.pkl', 'wb'))
pkl.dump(revalid_loader, open(f'./revalid_loader.pkl', 'wb'))

dl_env = IntegratedEnvironment(batch_shape[1], HIDDEN_SIZE, N_CLASSES, CNNet, OptimAgentDemoV2, DL_LR)
dl_env.set_loaders(train_loader, valid_loader)
dl_env.set_model()

adam_env = IntegratedEnvironment(batch_shape[1], HIDDEN_SIZE, N_CLASSES, CNNet, torch.optim.Adam, DL_LR)
adam_env.set_loaders(train_loader, valid_loader)
adam_env.set_model()

sgd_env = IntegratedEnvironment(batch_shape[1], HIDDEN_SIZE, N_CLASSES, CNNet, torch.optim.SGD, DL_LR)
sgd_env.set_loaders(train_loader, valid_loader)
sgd_env.set_model()

adagrad_env = IntegratedEnvironment(batch_shape[1], HIDDEN_SIZE, N_CLASSES, CNNet, torch.optim.Adagrad, DL_LR)
adagrad_env.set_loaders(train_loader, valid_loader)
adagrad_env.set_model()

rmsprop_env = IntegratedEnvironment(batch_shape[1], HIDDEN_SIZE, N_CLASSES, CNNet, torch.optim.RMSprop, DL_LR)
rmsprop_env.set_loaders(train_loader, valid_loader)
rmsprop_env.set_model()

rl_actor = PolicyNet(4, 256, ACTION_DIM).to(device)
rl_critic = CriticNet(4, 256).to(device)

rl_actor_optimizer = torch.optim.Adam(rl_actor.parameters(), lr = RL_LR)
rl_critic_optimizer = torch.optim.Adam(rl_critic.parameters(), lr = RL_LR)

eps = 1e-4
beta_1_min = 0.0 + eps
beta_1_max = 1.0 - eps

beta_2_min = 0.0 + eps
beta_2_max = 1.0 - eps

gamma_3_min = 0.0 + eps
gamma_3_max = 2.0 - eps

beta_1_action_map = np.arange(beta_1_min, beta_1_max, (beta_1_max - beta_1_min) / ACTION_DIM).astype(np.float32)
beta_2_action_map = np.arange(beta_2_min, beta_2_max, (beta_2_max - beta_2_min) / ACTION_DIM).astype(np.float32)
gamma_3_action_map = np.arange(gamma_3_min, gamma_3_max, (gamma_3_max - gamma_3_min) / ACTION_DIM).astype(np.float32)

episode_rewards = list()

for episode in range(MAX_EPISODES):

    benchmark = True if episode in benchmark_episode else False

    state, done = dl_env.init()

    if benchmark:
        torch.save(dl_env.state_dict(), './uniform_weights.pth')
        adam_env.init()
        sgd_env.init()
        adagrad_env.init()
        rmsprop_env.init()

        adam_env.load_state_dict(torch.load('./uniform_weights.pth'))
        sgd_env.load_state_dict(torch.load('./uniform_weights.pth'))
        adagrad_env.load_state_dict(torch.load('./uniform_weights.pth'))
        rmsprop_env.load_state_dict(torch.load('./uniform_weights.pth'))

    episode_reward = 0

    # INFO to save
    probs_1 = list()
    probs_2 = list()
    probs_3 = list()

    log_probs_1 = list()
    log_probs_2 = list()
    log_probs_3 = list()

    values = list()

    rewards = list()
    adam_rewards = list()
    sgd_rewards = list()
    adagrad_rewards = list()
    rmsprop_rewards = list()

    beta_1_list = list()
    beta_2_list = list()
    gamma_3_list = list()

    for epoch in tqdm(range(MAX_EPOCHS), f'{"||BENCHMARK||" if benchmark else ""} Episode {str(episode).zfill(3)} Training...'):
        # print(f'State: {state}, Done: {done}')
        state = state.to(device).view(1, -1)

        beta_1_probs, beta_2_probs, gamma_3_probs = rl_actor(state)
        beta_1_probs = beta_1_probs.squeeze()
        beta_2_probs = beta_2_probs.squeeze()
        gamma_3_probs = gamma_3_probs.squeeze()

        beta_1 = np.random.choice(np.arange(ACTION_DIM), p = beta_1_probs.detach().cpu().numpy())
        beta_2 = np.random.choice(np.arange(ACTION_DIM), p = beta_2_probs.detach().cpu().numpy())
        gamma_3 = np.random.choice(np.arange(ACTION_DIM), p = gamma_3_probs.detach().cpu().numpy())

        beta_1_list.append(beta_1)
        beta_2_list.append(beta_2)
        gamma_3_list.append(gamma_3)

        beta_1_log_prob = torch.log(beta_1_probs[beta_1])
        beta_2_log_prob = torch.log(beta_2_probs[beta_2])
        gamma_3_log_prob = torch.log(gamma_3_probs[gamma_3])

        beta_1_action = beta_1_action_map[beta_1]
        beta_2_action = beta_2_action_map[beta_2]
        gamma_3_action = gamma_3_action_map[gamma_3]

        value = rl_critic(state)

        dl_env.update_param(beta_1_action, beta_2_action, gamma_3_action)

        uni_train_loader, uni_valid_loader = random_sampler.get_dataloader(TRAIN_LEN, VALID_LEN)

        dl_env.replace_loader(uni_train_loader, uni_valid_loader)

        if benchmark:
            adam_env.replace_loader(uni_train_loader, uni_valid_loader)
            sgd_env.replace_loader(uni_train_loader, uni_valid_loader)
            adagrad_env.replace_loader(uni_train_loader, uni_valid_loader)
            rmsprop_env.replace_loader(uni_train_loader, uni_valid_loader)

        next_state, reward, done = dl_env.step(epoch)

        if benchmark:
            _, adam_reward, _ = adam_env.step(epoch)
            _, sgd_reward, _ = sgd_env.step(epoch)
            _, adagrad_reward, _ = adagrad_env.step(epoch)
            _, rmsprop_reward, _ = rmsprop_env.step(epoch)

        probs_1.append(beta_1_probs)
        probs_2.append(beta_2_probs)
        probs_3.append(gamma_3_probs)

        log_probs_1.append(beta_1_log_prob)
        log_probs_2.append(beta_2_log_prob)
        log_probs_3.append(gamma_3_log_prob)

        values.append(value)
        rewards.append(reward)

        if benchmark:
            adam_rewards.append(adam_reward)
            sgd_rewards.append(sgd_reward)
            adagrad_rewards.append(adagrad_reward)
            rmsprop_rewards.append(rmsprop_reward)

        beta_1_list.append(beta_1)
        beta_2_list.append(beta_2)
        gamma_3_list.append(gamma_3)

        state = next_state
        episode_reward += reward

    next_state = next_state.to(device)
    next_value = rl_critic(next_state).item()

    returns = compute_gae(rewards, [v.item() for v in values], next_value, DISCOUNT_FACTOR, GAE_LAMBDA)
    returns = torch.tensor(returns, dtype = torch.float32).to(device)

    values = torch.cat(values).squeeze()
    advantages = returns - values

    actor_loss_1 = -torch.sum(torch.stack(log_probs_1) * advantages.detach())
    actor_loss_2 = -torch.sum(torch.stack(log_probs_2) * advantages.detach())
    actor_loss_3 = -torch.sum(torch.stack(log_probs_3) * advantages.detach())
    actor_loss = (actor_loss_1 + actor_loss_2 + actor_loss_3) / 3

    rl_actor_optimizer.zero_grad()
    actor_loss.backward()
    rl_actor_optimizer.step()

    critic_loss = nn.functional.mse_loss(values, returns)
    rl_critic_optimizer.zero_grad()
    critic_loss.backward()
    rl_critic_optimizer.step()

    episode_rewards.append(episode_reward)

    if (episode + 1) % 10 == 0:
        print(f'Episode {episode + 1} / {MAX_EPISODES} | Reward: {episode_reward}')

    train_loss_dict = {
        'Adam': adam_env.train_loss,
        'SGD': sgd_env.train_loss,
        'Adagrad': adagrad_env.train_loss,
        'RMSprop': rmsprop_env.train_loss,
        'Ours': dl_env.train_loss
    }

    valid_loss_dict = {
        'Adam': adam_env.valid_loss,
        'SGD': sgd_env.valid_loss,
        'Adagrad': adagrad_env.valid_loss,
        'RMSprop': rmsprop_env.valid_loss,
        'Ours': dl_env.valid_loss
    }

    train_accuracy_dict = {
        'Adam': adam_env.train_accuracy,
        'SGD': sgd_env.train_accuracy,
        'Adagrad': adagrad_env.train_accuracy,
        'RMSprop': rmsprop_env.train_accuracy,
        'Ours': dl_env.train_accuracy
    }

    valid_accuracy_dict = {
        'Adam': adam_env.valid_accuracy,
        'SGD': sgd_env.valid_accuracy,
        'Adagrad': adagrad_env.valid_accuracy,
        'RMSprop': rmsprop_env.valid_accuracy,
        'Ours': dl_env.valid_accuracy
    }

    visualize(train_loss_dict, valid_loss_dict, train_accuracy_dict, valid_accuracy_dict, episode_rewards, episode)
    plt.savefig(f'./history/plots/mnist/episode_{episode}_plot.png')
    plt.close()

    torch.save(rl_actor.state_dict(), MODEL_PATH + f'actor/episode_{episode}_actor.pth')
    torch.save(rl_critic.state_dict(), MODEL_PATH + f'critic/episode_{episode}_critic.pth')

    dl_model = CNNet(INPUT_SIZE, HIDDEN_SIZE, N_CLASSES).to(device)
    dl_model.setup(input_shape=(32, 1, 28, 28), device=device)
    dl_optimizer = OptimAgentDemoV2(dl_model.parameters(), lr=3e-4)

    probs_1 = [val.cpu().detach().numpy() for val in probs_1]
    probs_2 = [val.cpu().detach().numpy() for val in probs_2]
    probs_3 = [val.cpu().detach().numpy() for val in probs_3]

    log_probs_1 = [val.item() for val in log_probs_1]
    log_probs_2 = [val.item() for val in log_probs_2]
    log_probs_3 = [val.item() for val in log_probs_3]

    values = [val.item() for val in values]

    episode_dict = dict(zip(episode_dict_keys, [
        probs_1, probs_2, probs_3,
        log_probs_1, log_probs_2, log_probs_3,
        values, rewards,
        beta_1_list, beta_2_list, gamma_3_list,
        train_loss_dict, valid_loss_dict,
        train_accuracy_dict, valid_accuracy_dict
    ]))

    pkl.dump(episode_dict, open(INFO_PATH + f'episode_{episode}_dict.pkl', 'wb'))