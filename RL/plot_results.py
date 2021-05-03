import csv 
import matplotlib.pyplot as plt
import numpy as np

class Ploter():
    def __init__(self, learning_rate, gamma, wall_rate):
        self.lr = learning_rate
        self.gamma = gamma
        self.wr = wall_rate

        path_test_raw = "Results/50%/0.5_1e-06_0.5_tilesVisited.csv"
        path_test_mean = "Results/50%/0.5_1e-06_0.5_tilesVisitedAverage.csv"
        fig_name = ["test0.png", "test1.png", "test2.png", "test3.png", "test4.png"]

        raw_test = self.load_from_file(path_test_raw)
        mean_test = self.load_from_file(path_test_mean)
        for i in range(5): # files contains 5 iterations 
            self.plot_bar_and_mean(raw_test[i], mean_test[i], fig_name[i])

        self.evolutive_mean_test = self.evolutive_mean(raw_test)
        
        #for i in range(5): # files contains 5 iterations 
        #    self.plot_bar_and_mean(raw_test[i], self.evolutive_mean_test[i], fig_name[i])

        
        #self.plot_mean_over_files(self.mean_over_files(mean_test), self.wr, "testMeanOverFiles.png")
        self.plot_mean_over_files(self.mean_over_files(self.evolutive_mean_test), self.wr, "new_version_50%.png")
        
        

    def plot_bar_and_mean(self, raw_lst, mean_lst, fig_name):
        x = [i for i in range(len(raw_lst))]
        
        plt.bar(x, raw_lst)
        plt.plot(x, mean_lst, color="tab:red")
        
        plt.title(f"Learning rate = {self.lr}, gamma = {self.gamma}")
        plt.xlabel("Epoque")
        plt.ylabel("Nombre de tiles parcourues")
        plt.savefig(fig_name)
        plt.cla()
        plt.clf()

    def plot_mean_over_files(self, mean_over_files_lst, wall_rate, fig_name):
        x = [i for i in range(len(mean_over_files_lst))]
         
        plt.axis((0,2000,0,280))
        plt.yticks(np.arange(0, 280, step=20))
        plt.plot(x, mean_over_files_lst, color="tab:red")
        plt.title(f"Moyenne sur 5 entraînements.\n Learning rate = {self.lr}, gamma = {self.gamma}, wall rate = {wall_rate}%")
        plt.xlabel("Epoque")
        plt.ylabel("Nombre de tiles parcourues")
        plt.savefig(fig_name)
        plt.cla()
        plt.clf()

    def evolutive_mean(self, raw_lst):
        ret = [[0 for i in range(2000)] for i in range(5)]
        for i in range(5):
            for j in range(1, 2001):
                ret[i][j-1] = sum(raw_lst[i][:j])/j
        
        return ret 



    def mean_over_files(self, files_lst):
        data = np.array(files_lst)
        return np.average(data, axis=0)
 

    def load_from_file(self, path):
        with open(path, newline='') as file:
            csv_reader = csv.reader(file)
            to_plot = []
            for row in csv_reader:
                to_plot.append(list(map(float,row)))

        return to_plot


if __name__ == '__main__':
    Ploter(0.000008, 0.99, 0.05)
    