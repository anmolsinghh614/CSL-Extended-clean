import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses):
    """
    Plot training and validation loss curves.

    Parameters:
    - train_losses: List of training losses for each epoch.
    - val_losses: List of validation losses for each epoch.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(train_accuracies, val_accuracies):
    """
    Plot training and validation accuracy curves.

    Parameters:
    - train_accuracies: List of training accuracies for each epoch.
    - val_accuracies: List of validation accuracies for each epoch.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_validation_accuracy(val_accuracies, num_epochs):
    """
    Plot validation accuracy over epochs.

    Parameters:
    - val_accuracies: List of validation accuracies for each epoch.
    - num_epochs: Total number of epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
