import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import time
import sys
import multiprocessing as mp
import os
import argparse
from ROOT import TFile, TTree
from array import array

SET_N_THREADS = 4

def findClusterDB(X, distance):
    if len(X)>1:
        db = DBSCAN(eps=distance, min_samples=2).fit(X)
        clusters = np.column_stack((X, db.labels_))
        nc = np.sum(clusters[:,2]==-1)
        Noise_inds=np.where(clusters[:,2]==-1)
        clusters=np.delete(clusters,Noise_inds,axis=0)
        labels=np.delete(db.labels_,Noise_inds,axis=0)
        unique, Areas = np.unique(labels, return_counts=True)
    elif len(X)==1:
        clusters=np.reshape([],(0,3))
        Areas=[]
        nc=1
    else:
        clusters=np.reshape([],(0,3))
        Areas=[]
        nc=0

    return clusters, Areas, nc


def par_findClusterDB(Total_packet, distance):
    input_data = []
    for packet in Total_packet:
        cd = (packet, distance)
        input_data.append(cd)

    pDB = mp.Pool(processes=SET_N_THREADS)
    DBresult = pDB.starmap(findClusterDB, input_data)
    pDB.close()
    pDB.join()

    return DBresult


def findClusterAgg(X, distance):
    if len(X) > 1:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, compute_full_tree=True).fit(X)
        clusters = np.column_stack((X, agg.labels_))
        unique, Areas = np.unique(agg.labels_, return_counts=True)
        Noise_inds = []
        Area_noise_inds = np.array(np.where(Areas == 1))
        if len(Area_noise_inds[0]) != 0:
            for index in Area_noise_inds[0]:
                Current_index = np.array(np.where(clusters[:, 2] == index))
                Noise_inds = np.append(Noise_inds, Current_index)
            Noise_inds = Noise_inds.astype(int)
            nc = len(Area_noise_inds[0])
            clusters = np.delete(clusters, Noise_inds, axis=0)
            labels = np.delete(agg.labels_, Noise_inds, axis=0)
            unique, Areas = np.unique(labels, return_counts=True)
        else:
            nc = 0
    elif len(X) == 1:
        clusters = np.reshape([], (0, 3))
        Areas = []
        nc = 1
    else:
        clusters = np.reshape([], (0, 3))
        Areas = []
        nc = 0

    return clusters, Areas, nc


def par_findClusterAgg(Total_packet, distance):
    input_data = []
    for packet in Total_packet:
        cd = (packet, distance,)
        input_data.append(cd)

    pAgg = mp.Pool(processes=SET_N_THREADS)
    Aggresult = pAgg.starmap(findClusterAgg, input_data)
    pAgg.close()
    pAgg.join()

    return Aggresult


class ACluster():  # analyzed cluster
    def __init__(self, mean, pca_r, area):
        self.mean = mean  # Cluster centroind
        self.pca_r = pca_r  # Principal axis variance ratio
        self.area = area  # Cluster area


def ClusterAnalysisPCA(Cluster):
    pca = PCA(n_components=2)
    pca.fit(Cluster)
    mean = np.array(pca.mean_)
    if pca.explained_variance_ratio_[1] != 0 and pca.explained_variance_ratio_[0] != 0:
        pca_r = min(10.0, (pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]))
    else:
        pca_r = 10.0
    area = len(Cluster)
    CC = ACluster(mean, pca_r, area)

    return CC


def par_ClusterAnalysisPCA(parallel_result):
    input_data = []
    for partial_res in parallel_result:
        unique_labels = np.unique(partial_res[0][:, 2], return_counts=False)
        for label in unique_labels:
            Index = np.where(partial_res[0][:, 2] == label)
            C = partial_res[0][Index]
            cd = (C[:, :2],)  # components with selected label
            input_data.append(cd)

    pPCA = mp.Pool(processes=SET_N_THREADS)
    PCAresult = pPCA.starmap(ClusterAnalysisPCA, input_data)
    pPCA.close()
    pPCA.join()

    return PCAresult


parser = argparse.ArgumentParser(description='Packet analyzer')
parser.add_argument('-f', '--Folder', metavar='N', type=str, default='Deafault_run',
                    help='Folder name to analyze')
parser.add_argument('-p', '--Par', action='store_true', dest='Par',
                    help='Enable parallelization')
parser.add_argument('-A', '--Alg', metavar='Agg or DB', type=str, default='DB',
                    help='Select which algorithm to use')
parser.add_argument('-d', '--Dist', metavar='N', type=int, nargs='+', default='[1,10]',
                    help='DB scan distance, Agglomerative cluster distance')


args = parser.parse_args()



if __name__=="__main__":
    Folder_name = args.Folder
    i = 0
    patience = 1
    ClusterDB = []
    AreaClusterDB = []
    ClusterAgg = []
    AreaClusterAgg = []

    noise_points = 0
    Data = []
    t=time.time()
    while True:
        file_name = args.Folder + "/" + args.Folder + "_packet_{0:0d}.npy".format(i)
        my_file = Path(file_name)
        if my_file.is_file():
            i += 1
        else:
            N_files=i
            print("{0:0d} Packet files found".format(N_files))
            break
    Ntot = i
    i = 0
    if args.Alg=="DB":
        if args.Par==True:
            print('Parallel processing with DBSCAN')
            for i in range(N_files):
                file_name = args.Folder + "/" + args.Folder + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    if (i % patience == 0):
                        sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        print("\nPacket {0:0d} is corrupted\n".format(i))
                        break
                    DBresult = par_findClusterDB(packet, args.Dist[0])
                    PCAresult = par_ClusterAnalysisPCA(DBresult)
                    Data = np.append(Data, PCAresult)
                    noise_points += sum(data[2] for data in DBresult)
                else:
                    print("\nNo more packets to analyze\n")
                    break
        else:
            print('Processing with DBSCAN')
            for i in range(N_files):
                file_name = args.Folder + "/" + args.Folder + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    if (i % patience == 0):
                        sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        print("\nPacket {0:0d} is corrupted\n".format(i))
                        break
                    for current_packet in packet:
                        Cluster, AreaCluster, nc = findClusterDB(current_packet,args.Dist[0])
                        for k in range(len(AreaCluster)):
                            Index=np.where(Cluster[:,2]==k)
                            C=Cluster[Index]
                            CC=ClusterAnalysisPCA(C[:,:2])
                            Data.append(CC)
                        noise_points  += nc
    elif args.Alg=="Agg":
        if args.Par==True:
            print('Parallel processing with Agglomerative Clustering')
            for i in range(N_files):
                file_name = args.Folder + "/" + args.Folder + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    if (i % patience == 0):
                        sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        print("\nPacket {0:0d} is corrupted\n".format(i))
                        break
                    Aggresult = par_findClusterAgg(packet, args.Dist[1])
                    PCAresult = par_ClusterAnalysisPCA(Aggresult)
                    Data = np.append(Data, PCAresult)
                    noise_points  += sum(data[2] for data in Aggresult)
                else:
                    print("\nNo more packets to analyze\n")
                    break
        else:
            print('Processing with Agglomerative Clustering')
            for i in range(N_files):
                file_name = args.Folder + "/" + args.Folder + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    if (i % patience == 0):
                        sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        print("\nPacket {0:0d} is corrupted\n".format(i))
                        break
                    for current_packet in packet:
                        Cluster, AreaCluster, nc = findClusterAgg(current_packet, args.Dist[1])
                        for k in range(len(AreaCluster)):
                            Index = np.where(Cluster[:, 2] == k)
                            C = Cluster[Index]
                            CC = ClusterAnalysisPCA(C[:, :2])
                            Data.append(CC)
                        noise_points  += nc

    else:
        print("Allora sei mona..")

    print('\n')
    areas = np.array([d.area for d in Data])
    ratios = np.array([d.pca_r for d in Data])
    means = np.array([d.mean for d in Data])
    print("Estimated noise points=", noise_points)
    print("Estimated clusters=", len(ratios))
    print("Estimated mean area=", np.mean(areas))
    print('Process time = ',time.time()-t)

    # get working dir
    path = os.getcwd()
    print("The current working directory is %s" % path)

    folder_path = path + '/Analyzed_Data/' + Folder_name
    try:  # new work folder named args.file_name
        os.mkdir(folder_path)
    except OSError:
        print("Directory %s alredy present" % folder_path)
    else:
        print("Successfully created the directory %s " % folder_path)
    String='Analyzed_Data/' + Folder_name + '/Area_' + args.Folder + '.npy'
    np.save(String, areas)
    String='Analyzed_Data/' + Folder_name + '/Mean_' + args.Folder + '.npy'
    np.save(String, means)
    String = 'Analyzed_Data/' + Folder_name + '/Ratio_' + args.Folder + '.npy'
    np.save(String, ratios)

    plt.subplot(311)
    plt.subplots_adjust(hspace=0.4)
    plt.title("Area")
    n, bins, patches = plt.hist(areas, bins=int((np.max(areas) - np.min(areas))/2),
                                range=(np.min(areas), np.max(areas)))
    plt.subplot(312)
    plt.title("PCA ratios")
    n, bins, patches = plt.hist(ratios, bins=int(np.max(ratios) - np.min(ratios)),
                                range=(np.min(ratios), np.max(ratios)))
    plt.subplot(313)
    plt.title("2D Histo")
    plt.hist2d(means[:, 1], means[:, 0], [128, 64])

    String = 'Analyzed_Data/' + Folder_name + '/' + args.Folder + '_plot'
    plt.savefig(String, dpi=700)

    String = 'Analyzed_Data/' + Folder_name + '/' + args.Folder + '.root'
    root_file = TFile(String, "RECREATE")
    tree = TTree("tree", "file")

    Rareas = array('f', [0.])
    Rmeanx = array('f', [0.])
    Rmeany = array('f', [0.])
    Rratios = array('f', [0.])

    tree.Branch("Rareas", Rareas, "Rareas/F")
    tree.Branch("Rmeanx", Rmeanx, "Rmeanx/F")
    tree.Branch("Rmeany", Rmeany, "Rmeany/F")
    tree.Branch("Rratios", Rratios, "Rratios/F")

    for i in range(len(areas)):
        Rareas[0] = areas[i]
        Rmeanx [0] = means[i,0]
        Rmeany[0] = means[i, 1]
        Rratios[0] = ratios[i]
        tree.Fill()

    root_file.Write()
    root_file.Close()
        