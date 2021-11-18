from GLOBALS import *
from alg_functions import *
import logging
"""
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
"""


class ALGPlotter:
    """
    This object is responsible for plotting, logging and neptune updating.
    """
    def __init__(self, plot_life=True, plot_neptune=False, name='', tags=None):

        if tags is None:
            tags = []
        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.run = {}
        self.name = name
        self.tags = tags
        self.fig, self.actor_losses, self.critic_losses, self.ax, self.agents_list = {}, {}, {}, {}, {}
        self.total_reward, self.val_total_rewards = [], []
        # self.prev_matrix_dict = {}
        self.matrix_dict = {}

        self.neptune_init()
        self.logging_init()

        if self.plot_life:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            self.data_to_plot = {}

        self.info("ALGPlotter instance created.")

    def neptune_init(self):
        if self.plot_neptune:
            self.run = neptune.init(project='1919ars/MountainCar', api_token=os.environ['NEPTUNE_API_TOKEN'],
                                    tags=self.tags, name=f'{self.name}_{ENV_NAME}',
                                    # source_files=['alg_constrants_amd_packages.py'],
                                    )

    def neptune_set_parameters(self, params_dict=None):
        if self.plot_neptune:
            params_dict = {
                f'{BATCH_SIZE}': BATCH_SIZE,
                f'{LR_CRITIC}': LR_CRITIC,
                f'{LR_ACTOR}': LR_ACTOR,
                f'{TAU}': TAU,
                f'{GAMMA}': GAMMA,
            }
            self.run['parameters'] = params_dict

    def neptune_plot(self, update_dict: dict):
        if self.plot_neptune:
            for k, v in update_dict.items():
                self.run[k].log(v)
                # self.run[k].log(f'{v}')

    def plots_update_data(self, data_dict, no_list=False):
        if self.plot_life:
            for key_name, value in data_dict.items():
                if no_list:
                    self.data_to_plot[key_name] = value
                else:
                    if key_name not in self.data_to_plot:
                        self.data_to_plot[key_name] = deque(maxlen=50000)
                    self.data_to_plot[key_name].append(value)

    def plots_online(self):
        # plot live:
        if self.plot_life:
            def plot_graph(ax, indx_r, indx_c, list_of_values, label, color='b', cla=True):
                if cla:
                    ax[indx_r, indx_c].cla()
                ax[indx_r, indx_c].plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                # ax[indx_r, indx_c].set_title(f'Plot: {label}')
                # ax[indx_r, indx_c].set_xlabel('iters')
                ax[indx_r, indx_c].set_ylabel(f'{label}')
                ax[indx_r, indx_c].axhline(0, color='gray')

            def plot_graph_axes(axes, list_of_values, label, color='b'):
                axes.cla()
                axes.plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                # axes.set_title(f'Plot: {label}')
                # axes.set_xlabel('iters')
                axes.set_ylabel(f'{label}')

            # counter = 0
            # for key_name, list_of_values in self.data_to_plot.items():
            #     plot_graph_axes(self.fig.axes[counter], list_of_values, key_name)
            #     counter += 1

            if 'rewards' in self.data_to_plot:
                plot_graph(self.ax, 1, 1, self.data_to_plot['rewards'], 'Rewards')

            plt.pause(0.05)

    def plot_nn_map(self, net_actor=None, net_critic=None, net_actor_target=None, net_critic_target=None):
        if self.plot_life:
            def plot_graph(ax, values, label, cla=True):
                if cla:
                    ax.cla()
                im = ax.imshow(values, cmap='hot', interpolation='nearest')
                # ax[indx_r, indx_c].set_title(f'Plot: {label}')
                # ax[indx_r, indx_c].set_xlabel('iters')
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # plt.colorbar(im, cax=cax)
                ax.set_ylabel(f'{label}')
                label = f'min: {np.min(values)}, max: {np.max(values)}'
                ax.set_xlabel(f'{label}')

            ax_list = self.fig.axes

            name = 'critic'
            current_mat = matrix_get(net_critic)
            prev_mat = self.matrix_get_prev(name)
            mat3 = current_mat - prev_mat
            plot_graph(ax_list[1], mat3, name)

            name = 'actor'
            current_mat = matrix_get(net_actor)
            prev_mat = self.matrix_get_prev(name)
            # print(f'(prev: {prev_mat[0][0]}, curr: {current_mat[0][0]})')
            mat3 = current_mat - prev_mat
            plot_graph(ax_list[0], mat3, name)
            plt.pause(0.05)

    def matrix_get_prev(self, name):
        if name not in self.matrix_dict:
            raise RuntimeError('Call matrix_update before get_prev!')
        return np.copy(self.matrix_dict[name])

    def matrix_update(self, name, net):
        matrix = matrix_get(net)
        self.matrix_dict[name] = np.copy(matrix)

    def close(self):
        if self.plot_neptune:
            self.run.stop()
        if self.plot_life:
            plt.close()

    @staticmethod
    def logging_init():
        # logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        # logging.basicConfig(level=logging.DEBUG)
        pass

    def info(self, message, print_info=True, end='\n'):
        # logging.info('So should this')
        if print_info:
            print(colored(f'~[INFO]: {message}', 'green'), end=end)

    def debug(self, message, print_info=True, end='\n'):
        # logging.debug('This message should go to the log file')
        if print_info:
            print(colored(f'~[DEBUG]: {message}', 'cyan'), end=end)

    def warning(self, message, print_info=True, end='\n'):
        # logging.warning('And this, too')
        if print_info:
            print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)

    def error(self, message):
        # logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
        raise RuntimeError(f"~[ERROR]: {message}")


# plotter = ALGPlotter(
#     plot_life=PLOT_LIVE,
#     plot_neptune=NEPTUNE,
#     name='my_run'
# )

