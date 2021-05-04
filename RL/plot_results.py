import csv 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

class Ploter():
    def __init__(self, learning_rate, gamma):
        self.lr = learning_rate
        self.gamma = gamma
        true_graph_data = []
        true_graph_data.append(self.plot_0())
        true_graph_data.append(self.plot_5())
        true_graph_data.append(self.plot_20())
        true_graph_data.append(self.plot_50())
        true_graph_data.append(self.plot_100())

        self.plot_true_graph(true_graph_data)
    
    def plot_0(self):
        path_raw = "Results/0%/0.0_1e-06_0.5_tilesVisited.csv"
        path_mean = "Results/0%/0.0_1e-06_0.5_tilesVisitedAverage.csv"
        fig_name = f"{self.lr}lr_{self.gamma}gamma_0percent_0.png"
            
        raw = self.load_from_file(path_raw)
        mean = self.load_from_file(path_mean)
        
        self.plot_bar_and_mean(raw, mean, fig_name, 0)

        return self.mean_over_files(mean)
        
    
    def plot_5(self):
        path_raw = "Results/5%/0.05_1e-06_0.5_tilesVisited.csv"
        path_mean = "Results/5%/0.05_1e-06_0.5_tilesVisitedAverage.csv"
        fig_name = f"{self.lr}lr_{self.gamma}gamma_5percent_0.png"

        raw = self.load_from_file(path_raw)
        mean = self.load_from_file(path_mean)

        self.plot_bar_and_mean(raw, mean, fig_name, 5)
        
        return self.mean_over_files(mean)

    def plot_20(self):
        path_raw = "Results/20%/0.2_1e-06_0.5_tilesVisited.csv"
        path_mean = "Results/20%/0.2_1e-06_0.5_tilesVisitedAverage.csv"
        fig_name = f"{self.lr}lr_{self.gamma}gamma_20percent_0.png"

        raw = self.load_from_file(path_raw)
        mean = self.load_from_file(path_mean)

        self.plot_bar_and_mean(raw, mean, fig_name, 20)
        
        return self.mean_over_files(mean) 
        
    def plot_50(self):
        path_raw = "Results/50%/0.5_1e-06_0.5_tilesVisited.csv"
        path_mean = "Results/50%/0.5_1e-06_0.5_tilesVisitedAverage.csv"
        fig_name = f"{self.lr}lr_{self.gamma}gamma_50percent_0.png"

        raw = self.load_from_file(path_raw)
        mean = self.load_from_file(path_mean)

        self.plot_bar_and_mean(raw, mean, fig_name, 50)
        
        return self.mean_over_files(mean)
    
    def plot_100(self):
        path_raw = "Results/100%/1.0_1e-06_0.5_tilesVisited.csv"
        path_mean = "Results/100%/1.0_1e-06_0.5_tilesVisitedAverage.csv"
        fig_name = f"{self.lr}lr_{self.gamma}gamma_100percent_0.png"

        raw = self.load_from_file(path_raw)
        mean = self.load_from_file(path_mean)
        
        self.plot_bar_and_mean(raw, mean, fig_name, 100)
       
        return self.mean_over_files(mean)

    def plot_true_graph(self, lst):
        self.plot_mean_over_files(lst)

    def plot_bar_and_mean(self, raw_lst, mean_lst, fig_name, wall_rate):
        x = [i for i in range(500)]
        
        fig = plt.figure(figsize=(8,5))
        grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.3)
        
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])
        ax4 = fig.add_subplot(grid[1, 0])
        ax5 = fig.add_subplot(grid[1, 1])

        ax1.bar(x, raw_lst[1][:500])
        ax1.plot(x, mean_lst[1][:500], color="tab:red")
        ax2.bar(x, raw_lst[2][:500])
        ax2.plot(x, mean_lst[2][:500], color="tab:red")
        ax3.bar(x, raw_lst[3][:500])
        ax3.plot(x, mean_lst[3][:500], color="tab:red")
        ax4.bar(x, raw_lst[4][:500])
        ax4.plot(x, mean_lst[4][:500], color="tab:red")
        ax5.bar(x, raw_lst[0][:500])
        ax5.plot(x, mean_lst[0][:500], color="tab:red")
        
        fig.suptitle(f"Nombre de tiles parcourues par les seed 0, 1, 2, 3 et 4 \nLearning rate = {self.lr}, gamma = {self.gamma}, wall rate = {wall_rate}%")
        
        plt.savefig(fig_name)
        plt.cla()
        plt.clf()

    def plot_mean_over_files(self, mean_over_files_lst):
        x = [i for i in range(len(mean_over_files_lst[0]))]
         
        plt.axis((0,500,0,280))
        plt.grid(True)
        plt.yticks(np.arange(0, 300, step=20))
        plt.plot(x, mean_over_files_lst[0], color="tab:red", label="0%")
        plt.plot(x, mean_over_files_lst[1], color="tab:green", label="5%")
        plt.plot(x, mean_over_files_lst[2], color="tab:blue", label="20%")
        plt.plot(x, mean_over_files_lst[3], color="tab:orange", label="50%")
        plt.plot(x, mean_over_files_lst[4], color="tab:purple", label="100%")
        plt.title(f"Comparaison des paramètres d'apparitions de murs. \n Moyenne sur 5 entraînements (learning rate = {self.lr}, gamma = {self.gamma})")
        plt.xlabel("Epoque")
        plt.ylabel("Nombre de tiles parcourues")
        
        plt.legend(bbox_to_anchor=(0.85, 0.75), loc='upper left', borderaxespad=0.)

        plt.savefig("overall_results.png")
        plt.cla()
        plt.clf()

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
    Ploter(0.000001, 0.50)
    