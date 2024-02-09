import matplotlib.pyplot as plt


def save_scatterplot(duration, residuals, plot_path=None, save=True, show=False):
    plt.figure(figsize=(10, 6))
    plt.scatter(duration, residuals["deviance"])
    plt.title('Scatter plot of deviance residuals')
    plt.xlabel("Duration")
    plt.ylabel('Deviance residuals')
    if save:
        plt.savefig(plot_path)
    if show:
        plt.show()
    plt.close()
