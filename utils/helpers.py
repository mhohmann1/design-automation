import matplotlib.pyplot as plt

def plot_loss(train_loss, test_loss, label, save_img=False, show_img=False, path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label=f"Training {label}")
    plt.plot(test_loss, label=f"Testing {label}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{label}")
    plt.legend(loc="upper right")
    if save_img:
        plt.savefig(path)
    if show_img:
        plt.show()
    plt.close()
