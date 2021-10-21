import numpy as np
import matplotlib.pyplot as plt

# def save_loss_history(train_loss, test_loss, epochs):
#     total_epochs = np.linspace(0, epochs-1, epochs)
#     fig, ax = plt.subplots(figsize=(12,8))
#     plt.xlabel('Epoch', fontsize=15)
#     plt.ylabel('Loss', fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     ax.plot(total_epochs, np.array(train_loss), label='Train')
#     ax.plot(total_epochs, np.array(test_loss), label='Validation')
#     ax.legend(prop={'size': 15})
#     # plt.show()
#     plt.savefig('output/loss.png')

# def save_loss_history(train_loss, epochs):
#     total_epochs = np.linspace(0, epochs-1, epochs)
#     fig, ax = plt.subplots(figsize=(12,8))
#     plt.xlabel('Epoch', fontsize=15)
#     plt.ylabel('Loss', fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     ax.plot(total_epochs, np.array(train_loss), label='Train')
#     ax.legend(prop={'size': 15})
#     # plt.show()
#     plt.savefig('output/loss.png')

# def save_accuracy_history(train_acc, test_acc, epochs):
#     total_epochs = np.linspace(0, epochs-1, epochs)
#     fig, ax = plt.subplots(figsize=(12,8))
#     plt.xlabel('Epoch', fontsize=15)
#     plt.ylabel('Accuracy', fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     ax.plot(total_epochs, np.array(train_acc), label='Train')
#     ax.plot(total_epochs, np.array(test_acc), label='Validation')
#     ax.legend(prop={'size': 15})
#     # plt.show()
#     plt.savefig('output/acc.png')

# def save_top3error_history(test_acc, epochs):
#     total_epochs = np.linspace(0, epochs-1, epochs)
#     fig, ax = plt.subplots(figsize=(12,8))
#     plt.xlabel('Epoch', fontsize=15)
#     plt.ylabel('Top3 Error Rate', fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     ax.plot(total_epochs, np.array(test_acc), label='Validation')
#     ax.legend(prop={'size': 15})
#     # plt.show()
#     plt.savefig('output/val_top3error.png')


def majorityVote(models_preds):
    final_pred = []

    models_preds_array = np.array(models_preds).T
    for row in models_preds_array:
        counts = np.bincount(row, minlength=2)
        predicted_class = np.argmax(counts)
        final_pred.append(predicted_class)

    return final_pred