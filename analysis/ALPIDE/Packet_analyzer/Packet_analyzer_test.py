import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from pathlib import Path
from matplotlib import cm
import matplotlib.colors as colors
import time
import sys
import multiprocessing as mp
import os
import argparse
from tqdm import tqdm
from ROOT import TFile, TTree


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

def Verbose_print(String,Verb=True):
    if Verb:
        print(String)



parser = argparse.ArgumentParser(description='Packet analyzer')
parser.add_argument('-Df', '--Data_path', metavar='N', type=str, default='Data',
                    help='Data folder path')
parser.add_argument('-f', '--Folder', metavar='N', type=str, default='Deafault_run',
                    help='Folder name to analyze')
parser.add_argument('-p', '--Par', action='store_true', dest='Par',
                    help='Enable parallelization')
parser.add_argument('-v', '--Verbose', action='store_true', dest='Verbose',
                    help='Enable verbose option')
parser.add_argument('-A', '--Alg', metavar='Agg or DB', type=str, default='DB',
                    help='Select which algorithm to use')
parser.add_argument('-d', '--Dist', metavar='N', type=int, nargs='+', default=[1,10],
                    help='DB scan distance, Agglomerative cluster distance')
parser.add_argument('-s', '--StepCenter', metavar='N', type=int, default='17',
                    help='Distance from the center, in motor steps')
#parser.add_argument('-w', '--Weight', metavar='N', type=float, default='1.',
                    #help="Event's weight")
parser.add_argument('-NThr', '--NThreads', metavar='N', type=int, default='4',
                    help="Number of threahds")
parser.add_argument('-b', '--NBestemmie', metavar='N', type=int, default='1000',
                    help="Analysis's complexity")

args = parser.parse_args()


SET_N_THREADS = args.NThreads

#Geometry parameters
ALPIDE_center   = -17   #Steps
ALPIDE_distance = 78.63 #mm
ALPIDE_width    = 30    #mm
ALPIDE_height   = 13.8    #mm



if __name__=="__main__":
                    
    Data_folder_path = args.Data_path                
    Folder_name      = args.Folder
    
    #patience = 1                
    i              = 0
    ClusterDB      = []
    AreaClusterDB  = []
    ClusterAgg     = []
    AreaClusterAgg = []

    #find number of files
    noise_points = 0
    Data         = []
    t            = time.time()
    while True:
        file_name =Data_folder_path + "/" + Folder_name + "/" + Folder_name + "_packet_{0:0d}.npy".format(i)
        my_file = Path(file_name)
        if my_file.is_file():
            i += 1
        else:
            N_files=i
            Verbose_print("{0:0d} Packet files found".format(N_files),args.Verbose)
            break
    Ntot = i
    i = 0

    #clustering and analysis
    if args.Alg=="DB":
        if args.Par==True:
            Verbose_print('Parallel processing with DBSCAN',args.Verbose)
            for i in tqdm(range(N_files), desc=args.Folder):
                file_name =Data_folder_path + "/" + Folder_name + "/" + Folder_name + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    #if (i % patience == 0):
                        #sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        Verbose_print("\nPacket {0:0d} is corrupted\n".format(i),args.Verbose)
                        break
                    DBresult = par_findClusterDB(packet, args.Dist[0])
                    PCAresult = par_ClusterAnalysisPCA(DBresult)
                    Data = np.append(Data, PCAresult)
                    noise_points += sum(data[2] for data in DBresult)
                else:
                    Verbose_print("\nNo more packets to analyze\n",args.Verbose)
                    break
        else:
            Verbose_print('Processing with DBSCAN',args.Verbose)
            for i in tqdm(range(N_files),desc=args.Folder):
                file_name =Data_folder_path + "/" + Folder_name + "/" + Folder_name + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    #if (i % patience == 0):
                        #sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        Verbose_print("\nPacket {0:0d} is corrupted\n".format(i),args.Verbose)
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
            Verbose_print('Parallel processing with Agglomerative Clustering',args.Verbose)
            for i in tqdm(range(N_files),desc=args.Folder):
                file_name =Data_folder_path + "/" + Folder_name + "/" + Folder_name + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    #if (i % patience == 0):
                        #sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        Verbose_print("\nPacket {0:0d} is corrupted\n".format(i),args.Verbose)
                        break
                    Aggresult = par_findClusterAgg(packet, args.Dist[1])
                    PCAresult = par_ClusterAnalysisPCA(Aggresult)
                    Data = np.append(Data, PCAresult)
                    noise_points  += sum(data[2] for data in Aggresult)
                else:
                    Verbose_print("\nNo more packets to analyze\n",args.Verbose)
                    break
        else:
            Verbose_print('Processing with Agglomerative Clustering',args.Verbose)
            for i in tqdm(range(N_files),desc=args.Folder):
                file_name =Data_folder_path + "/" + Folder_name + "/" + Folder_name + "_packet_{0:0d}.npy".format(i)
                my_file = Path(file_name)
                if my_file.is_file():
                    #if (i % patience == 0):
                        #sys.stdout.write("\rCompleteness : " + str(round(i / Ntot * 100, 1)) + "%")
                    try:
                        packet = np.load(file_name, allow_pickle=True)
                    except:
                        Verbose_print("\nPacket {0:0d} is corrupted\n".format(i),args.Verbose)
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
        Verbose_print("Allora sei mona..",args.Verbose)

    #print results
    Verbose_print('\n',args.Verbose)
    areas = np.array([d.area for d in Data])
    ratios = np.array([d.pca_r for d in Data])
    means = np.array([d.mean for d in Data])
    if args.Verbose:
        print("Estimated noise points=", noise_points)
        print("Estimated clusters=", len(ratios))
        print("Estimated mean area=", np.mean(areas))
        print('Process time = ',time.time()-t)

    # get working dir
    path = os.getcwd()
    Verbose_print("The current working directory is %s" % path,args.Verbose)

    #save histos as png
    folder_path = path + '/Analyzed_Data/' + Folder_name
    try:  # new work folder named args.file_name
        os.mkdir(folder_path)
    except OSError:
        Verbose_print("Directory %s alredy present" % folder_path,args.Verbose)
    else:
        Verbose_print("Successfully created the directory %s " % folder_path,args.Verbose)
    String='Analyzed_Data/' + Folder_name + '/Area_' + args.Folder + '.npy'
    np.save(String, areas)
    String='Analyzed_Data/' + Folder_name + '/Mean_' + args.Folder + '.npy'
    np.save(String, means)
    String = 'Analyzed_Data/' + Folder_name + '/Ratio_' + args.Folder + '.npy'
    np.save(String, ratios)

    plt.subplot(311)
    plt.subplots_adjust(hspace=0.4)
    plt.title("Area")
    try:
        n, bins, patches = plt.hist(areas, bins=int((np.max(areas) - np.min(areas))),
                                    range=(np.min(areas), np.max(areas)))
    except:
        n, bins, patches = plt.hist(areas)
    plt.subplot(312)
    plt.title("PCA ratios")
    try:
        n, bins, patches = plt.hist(ratios, bins=int(np.max(ratios) - np.min(ratios)),
                                    range=(np.min(ratios), np.max(ratios)))
    except:
        n, bins, patches = plt.hist(ratios)
    plt.subplot(313)
    plt.title("2D Histo")
    plt.hist2d(means[:, 1], means[:, 0], [128, 64])

    String = 'Analyzed_Data/' + Folder_name + '/' + args.Folder + '_plot'
    plt.savefig(String, dpi=700)
    plt.close()

    #produce an hitmap matrix image
    cluster_matrix = np.zeros((512, 1024))
    for i in range(N_files):
        file_name =Data_folder_path + "/" + Folder_name + "/" + Folder_name + "_packet_{0:0d}.npy".format(i)
        packet = np.load(file_name, allow_pickle=True)

        for Cluster in packet:
            for pixel in Cluster:
                x = pixel[1]
                y = pixel[0]
                cluster_matrix[y, x] += 1

    # and plot it
    fig, ax = plt.subplots()
    colormap = cm.get_cmap('jet')
    psm = ax.pcolormesh(cluster_matrix, cmap=colormap, norm=colors.LogNorm(vmin=1))
    # psm = ax.pcolormesh(cluster_matrix, cmap=colormap, vmax=50)
    cbar = plt.colorbar(psm, shrink=0.5, ax=ax)
    # cbar.set_label("N. hits")
    plt.axis('scaled')
    ax.set(xlim=(0, 1023), ylim=(0, 511))
    plt.xlabel("Column")
    plt.ylabel("Row")
    String = 'Analyzed_Data/' + Folder_name + '/' + args.Folder + '_hitmap'
    plt.savefig(String, dpi=700)
    plt.close()


    #save data as root file
    String      = 'Analyzed_Data/' + Folder_name + '/' + args.Folder + '.root'
    root_file   = TFile(String, "RECREATE")
    tree        = TTree("tree", "file")
    
    file=open(Data_folder_path+"/"+Folder_name+"/Stat_"+Folder_name+".txt")
    line=file.readlines()
    for l in line:
        if (l.startswith("Readout_time")):
            time=float(l.split('=')[1].strip()[:-1])
        if (l.startswith("Strobe")):
            Strobe=int(l.split('=')[1].strip()[:-1])
        if (l.startswith("Gap")):
             Gap=int(l.split('=')[1].strip()[:-1])
    file.close()

    Rnoise  = np.empty(1, dtype="float32")
    Rareas  = np.empty(1, dtype="float32")
    Rmeanx  = np.empty(1, dtype="float32")
    Rmeany  = np.empty(1, dtype="float32")
    Rtheta  = np.empty(1, dtype="float32")
    #Rweight = np.empty(1, dtype="float32")
    Reetime = np.empty(1, dtype="float32")
    Rratios = np.empty(1, dtype="float32")

    tree.Branch("Rnoise",  Rnoise,  "Rnoise/F")
    tree.Branch("Rareas",  Rareas,  "Rareas/F")
    tree.Branch("Rmeanx",  Rmeanx,  "Rmeanx/F")
    tree.Branch("Rmeany",  Rmeany,  "Rmeany/F")
    tree.Branch("Rtheta",  Rtheta,  "Rtheta/F")
    #tree.Branch("Rweight", Rweight, "Rweight/F")
    tree.Branch("Reetime", Reetime, "Reetime/F")
    tree.Branch("Rratios", Rratios, "Rratios/F")

    for i in range(len(areas)):
        Rnoise[0]   = noise_points
        Rareas[0]   = areas[i]
        Rmeany [0]  = means[i, 0]
        Rmeanx[0]   = means[i, 1]
        Rtheta[0]   = np.arctan2((means[i, 1] - 547)*ALPIDE_width/1024,
                                 np.sqrt(((means[i, 0] - 251)*ALPIDE_height/512)**2+(ALPIDE_distance)**2))+ (args.StepCenter+ALPIDE_center)*0.9*np.pi/180
        #Rtheta[0]   = np.arctan2((means[i, 1] - 512)*ALPIDE_width/1024,
                               #ALPIDE_distance) + (args.StepCenter+ALPIDE_center)*0.9*np.pi/180    
        #Rweight[0]  = args.Weight
        Reetime[0]  = (Strobe*time)/(Gap+Strobe) #effective exposition time [s]
        Rratios[0]  = ratios[i]
        tree.Fill()

    root_file.Write()
    root_file.Close()
        
