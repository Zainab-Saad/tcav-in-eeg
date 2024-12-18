import matplotlib.pyplot as plt
import os

# # Create a folder to save plots
# output_folder = "plots"
# os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

def plot_loss(epochs, train_loss, valid_loss, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_name = "training_validation_loss.png"
    output_path = os.path.join(output_dir, plot_name)
    plt.savefig(output_path, dpi=300)  # Save plot with 300 DPI for good quality
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to: {output_path}")


def plot_accuracy(epochs, train_acc, valid_acc, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, valid_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_name = "training_validation_accuracy.png"
    output_path = os.path.join(output_dir, plot_name)
    plt.savefig(output_path, dpi=300)  # Save plot with 300 DPI for good quality
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to: {output_path}")