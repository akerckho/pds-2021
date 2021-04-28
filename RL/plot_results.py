import csv 
import matplotlib.pyplot as plt

class Ploter(object):
    def __init__(self, learning_rate, gamma):
        self.lr = learning_rate
        self.gamma = gamma

        path_test_raw = "Results/0.05_tilesVisited.csv"
        path_test_mean = "Results/0.05_tilesVisitedAverage.csv"
        fig_name = ["test0.png", "test1.png", "test2.png", "test3.png", "test4.png"]

        raw_test = self.load_from_file(path_test_raw)
        mean_test = self.load_from_file(path_test_mean)
        for i in range(1): # files contains 5 iterations 
            self.plot_bar_and_mean(raw_test[i], mean_test[i], fig_name[i])

    def plot_bar_and_mean(self, raw_lst, mean_lst, fig_name):
        x = [i for i in range(len(raw_lst))]
        
        plt.bar(x, raw_lst)
        plt.plot(x, mean_lst, color="tab:red")
        
        plt.title(f"Learning rate = {self.lr}, gamma = {self.gamma}")
        plt.xlabel("Epoque")
        plt.ylabel("Nombre de tiles parcourues")
        plt.savefig(fig_name)

    def mean_over_files(self, files_lst):
        pass

    def load_from_file(self, path):
        with open(path, newline='') as file:
            csv_reader = csv.reader(file)
            to_plot = []
            for row in csv_reader:
                to_plot.append(row)

        return to_plot


if __name__ == '__main__':
    Ploter(0.000008, 0.99)
    