from dataloader import *
import utils

if __name__ == '__main__':
    NotImplemented
    # from ckpt to spectrogram

    ax = plt.subplot(gs[0])
    # ax.figure.set_size_inches(18,10)
    ax = plt.plot(data['x'].values[120:], color='25', label='Raw Siganl')

    ax = plt.plot(ensemble_prediction_x_a_s.mean().values, color='green', label='ANN')
    ax = plt.plot(ensemble_prediction_x_l_s.mean().values, color='blue', label='PHTNet')
    ax = plt.plot(ensemble_prediction_x_n_s.mean().values, color='black', label='our method')
    ax = plt.plot(ground_truth_x, color='red', label='voluntary motion')
    ax2 = plt.legend(loc='upper right')

    # plt.legend()
    ax2 = plt.subplot(gs[1])
    # ax2.figure.set_size_inches(18,5)
    ax2 = plt.plot((ground_truth_x - ensemble_prediction_x_l_s.mean()), color='green', label='ANN')
    ax2 = plt.plot((ground_truth_x - ensemble_prediction_x_a_s.mean()), color='red', label='PHTnet')
    ax2 = plt.plot((ground_truth_x - ensemble_prediction_x_n_s.mean()), color='black', label='Our method')
    ax2 = plt.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('fig1.png', dpi=300)
