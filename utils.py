import dgl
import numpy as np
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from main import args_define
import networkx as nx
import ast
import en_core_web_lg
from community import *
from DRL import *

args = args_define.args
# Dataset
def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features

class SocialDataset(Dataset):
    def __init__(self, path, index):
        new_f_path = path #'/home/XXXX-1/Paper3/KPGNN/FinEvent/incremental' -->commented on April 30 for checking Event2018 performance
        self.features = np.load(new_f_path + '/' + str(index) + '/features.npy')
        # print("PATH:",path)
        #XXXX-1 - community features
        #print("DEBUG:",self.features.shape)
        self.nlp = en_core_web_lg.load()
        message_df = pd.read_csv(path+'/' + str(index) +'/df.csv',lineterminator='\n')
        if args.use_community:
            # Find community
            self.s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
            # print("DEBUG2:",self.s_bool_A_tid_tid.shape)
            #g = nx.Graph(s_bool_A_tid_tid)
            self.adj = self.s_bool_A_tid_tid.todense()
            # print("DEBUG2.1:",self.adj.shape)
            self.rows, self.cols = np.where( self.adj == 1)
            #print("Rows:",self.rows)
            self.edges = list(zip(self.rows.tolist(), self.cols.tolist()))
            #print(self.edges)
            #print()
            #edges = np.array(list(g.edges()))
            self.comm = def_comm(self.edges)
            print("Community detection done")
            self.rev_com = {j: i for i in self.comm.keys() for j in self.comm[i]}
            # print("Max community number :",max(self.rev_com.values()),len(self.comm.keys()))
            # print(self.rev_com)
            # print(len(self.rev_com))
            indices = np.load(path+'/' + str(index) +'/indices.npy') #node_tweet_map
            extra_info = [self.rev_com[i] for i in range(0,len(self.rev_com.keys()))]
            message_df['add']=[i+[str(j)] for i,j in zip(message_df.find_filtered_words.apply(eval).to_list(),extra_info)]
            # print(message_df['add'])
            # run next line if community info is needed
            self.features = message_df['add'].apply(lambda x: self.nlp((' ').join(x)).vector).values
        else:
            # run next two lines if community info is not needed
            message_df.filtered_words = message_df.find_filtered_words.apply(eval)
            self.features = message_df.find_filtered_words.apply(lambda x: self.nlp(' '.join(x)).vector).values
        
        self.features = np.stack(self.features, axis=0)
        t_features = df_to_t_features(message_df)
        self.features = np.concatenate((self.features, t_features), axis=1)
        print("DEBUG:",self.features.shape)



        '''
        self.new_fe = np.array([self.nlp(str(i)).vector for i in self.rev_com.values()]) # sort

        # #strategy encode features here
        # df = pd.read_csv('/home/XXXX-1/Paper3/KPGNN/datasets/Twitter/df.csv',index=True)
        # df = df[index==]
        # features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
        # self.features = np.stack(features, axis=0)

        # print("New features", self.new_fe.shape)
        # print("Original features: ",self.features.shape)
        self.features = np.concatenate((self.features,self.new_fe),axis=1)
        # print("DEBUG3:",len(self.features))
        '''
        # Use this path for using community as labels
        new_labels_path = path#'/home/XXXX-1/Paper3/KPGNN_EVENT/KPGNN/incremental_train'#'/home/XXXX-1/Paper3/KPGNN/FinEvent/incremental'#, -->commented on May 1 for checking Event2018 performance
        print("new_label_path", new_labels_path + '/' + str(index) + '/labels.npy')
        temp = np.load(new_labels_path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        # Commented on 1 may 2025, temp = np.load(new_labels_path + '/' + str(index) + '/label_new_4.npy', allow_pickle=True) # to use community as labels
        #temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True) # to use original labels
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix, self.matrix1, self.matrix2, self.matrix3 = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        
        # XXXX-1
        # entity
        s_tid_tid_entity = sparse.load_npz(path + '/' + str(index) + '/s_m_tid_entity_tid.npz')
        s_bool_A_tid_entity_tid = s_tid_tid_entity.astype('bool')
        # words
        '''
        For the folder: /home/XXXX-1/Paper3/KPGNN_attn/KPGNN/incremental_filtered_French/
        The files s_m_tid_word_tid.npz are empty. So I have replaced them with hash links from /home/XXXX-1/Paper3/CLKD/datasets/318_ALL_French/
        '''
        if 'french' in path.lower():
            s_tid_tid_hash = sparse.load_npz('/home/XXXX-1/Paper3/CLKD/datasets/French/' + '/' + str(index) + '/s_m_tid_hash_tid.npz')
            s_bool_A_tid_hash_tid = s_tid_tid_hash.astype('bool')
        else:
            s_tid_tid_word = sparse.load_npz(path + '/' + str(index) + '/s_m_tid_word_tid.npz')
            s_bool_A_tid_word_tid = s_tid_tid_word.astype('bool')
        # user_id
        s_tid_tid_userid = sparse.load_npz(path + '/' + str(index) + '/s_m_tid_userid_tid.npz')
        s_bool_A_tid_userid_tid = s_tid_tid_userid.astype('bool')
        print("My: All three graphs loaded")
        if 'french' in path.lower():
            return s_bool_A_tid_tid, s_bool_A_tid_entity_tid, s_bool_A_tid_hash_tid, s_bool_A_tid_userid_tid
        return s_bool_A_tid_tid, s_bool_A_tid_entity_tid, s_bool_A_tid_word_tid, s_bool_A_tid_userid_tid

    # Used by remove_obsolete mode 1
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]  # keep row
            self.matrix = self.matrix[:, indices_to_keep]  # keep column
            #  remove nodes from matrix


# Compute the representations of all the nodes in g using model
def extract_embeddings_old(g, model, num_all_samples, labels):
    with torch.no_grad():
        model.eval()
        for batch_id, nf in enumerate(
                dgl.contrib.sampling.NeighborSampler(g,  # sample from the whole graph (contain unseen nodes)
                                                     num_all_samples,  # set batch size = the total number of nodes
                                                     1000,
                                                     # set the expand_factor (the number of neighbors sampled from
                                                     # the neighbor list of a vertex) to None: get error: non-int
                                                     # expand_factor not supported
                                                     neighbor_type='in',
                                                     shuffle=False,
                                                     num_workers=32,
                                                     num_hops=2)):
            nf.copy_from_parent()
            if args.use_dgi:
                extract_features, _ = model(nf)  # representations of all nodes
            else:
                extract_features = model(nf)  # representations of all nodes
            extract_nids = nf.layer_parent_nid(-1).to(device=extract_features.device, dtype=torch.long)  # node ids
            extract_labels = labels[extract_nids]  # labels of all nodes
        assert batch_id == 0
        extract_nids = extract_nids.data.cpu().numpy()
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        assert (A == extract_nids).all()

    return extract_nids, extract_features, extract_labels

def extract_embeddings(g, g_2, g_3, g_4, model, num_all_samples, labels):
    with torch.no_grad():
        model.eval()
        samples = dgl.contrib.sampling.NeighborSampler(g,
                                                            num_all_samples,
                                                            1000,
                                                            neighbor_type='in',
                                                            shuffle=False,
                                                            num_workers=32,
                                                            num_hops=2)
        samples_1 = dgl.contrib.sampling.NeighborSampler(g_2,
                                                            num_all_samples,
                                                            1000,
                                                            neighbor_type='in',
                                                            shuffle=False,
                                                            num_workers=32,
                                                            num_hops=2)
        samples_2 = dgl.contrib.sampling.NeighborSampler(g_3,
                                                            num_all_samples,
                                                            1000,
                                                            neighbor_type='in',
                                                            shuffle=False,
                                                            num_workers=32,
                                                            num_hops=2)
        samples_3 = dgl.contrib.sampling.NeighborSampler(g_4,
                                                            num_all_samples,
                                                            1000,
                                                            neighbor_type='in',
                                                            shuffle=False,
                                                            num_workers=32,
                                                            num_hops=2)
        for (batch_id, nf), (batch_id2, nf1), (batch_id3, nf2), (batch_id4, nf3) in zip(enumerate(samples), enumerate(samples_1), enumerate(samples_2), enumerate(samples_3)):
            nf.copy_from_parent()
            nf1.copy_from_parent()
            nf2.copy_from_parent()
            nf3.copy_from_parent()
            if args.use_dgi:
                extract_features, _ = model(nf)  # representations of all nodes
            else:
                extract_features = model(nf, nf1, nf2, nf3)  # representations of all nodes
            extract_nids = nf.layer_parent_nid(-1).to(device=extract_features.device, dtype=torch.long)  # node ids
            extract_labels = labels[extract_nids]  # labels of all nodes
        assert batch_id == 0
        extract_nids = extract_nids.data.cpu().numpy()
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        assert (A == extract_nids).all()

    return extract_nids, extract_features, extract_labels

def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")
    print()


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()
    #print("DEBUG: K-Means",extract_labels.shape)
    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans clustering NMI
    return (n_test_tweets, n_classes, nmi, ami, ari)


def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices)
    #n_tweets, n_classes, nmi, ami, ari = run_drldbscan(extract_features, extract_labels, indices)

    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' NMI: '
    message += str(nmi)
    message += '\n\t' + mode + ' AMI: '
    message += str(ami)
    message += '\n\t' + mode + ' ARI: '
    message += str(ari)

    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices,
                                              save_path + '/isolated_nodes.pt')
        '''
        n_tweets, n_classes, nmi, ami, ari = run_drldbscan(extract_features, extract_labels, indices,
                                              save_path + '/isolated_nodes.pt')
        '''
        
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' NMI: '
        message += str(nmi)
        message += '\n\t' + mode + ' AMI: '
        message += str(ami)
        message += '\n\t' + mode + ' ARI: '
        message += str(ari)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    return nmi, ami, ari


def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    #XXXX-1
    i_n = [i for i in G.nodes() if G.in_degree(i) == 0]
    #XXXX-1
    #print("Isolated nodes: ", i_n)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes



def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, num_indices_to_remove=0):
    """
        Intro:
        This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
        If remove_obsolete mode 0 or 1:
        For initial/maintenance epochs:
        - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set
        Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, only their features and structural info are leveraged)
        If remove_obsolete mode 2:
        For initial/maintenance epochs:
        - The (i + 1) = (train_i + 1)th block (block train_i = i) is used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set

        :param length: the length of label list
        :param data_split: loaded splited data (generated in custom_message_graph.py)
        :param train_i, i: flag, indicating for initial/maintenance stage if train_i == i and inference stage for others
        :param validation_percent: the percent of validation data occupied in whole dataset
        :param save_path: path to save data
        :param num_indices_to_remove: number of indices ought to be removed

        :returns train indices, validation indices or test indices
    """
    if args.remove_obsolete == 0 or args.remove_obsolete == 1:  # remove_obsolete mode 0 or 1
        # verify total number of nodes
        assert length == (np.mean(data_split[:i + 1]) - num_indices_to_remove)

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the training indices
            train_length = np.sum(data_split[:train_i + 1])
            train_length -= num_indices_to_remove
            train_indices = torch.randperm(int(train_length))
            # get total number of validation indices
            n_validation_samples = int(train_length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path + '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If the process is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
            test_indices += (np.sum(data_split[:i]) - num_indices_to_remove)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices

    else:  # remove_obsolete mode 2
        # verify total number of nodes
        assert length == data_split[i]

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the graph indices
            train_indices = torch.randperm(length)
            # get total number of validation indices
            n_validation_samples = int(length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path +
                           '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(
                    save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(
                0, (data_split[i] - 1), dtype=torch.long)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices


# Utility function, finds the indices of the values' elements in tensor
def find(tensor, values):
    return torch.nonzero(tensor.cpu()[..., None] == values.cpu())

def run_drldbscan(extract_features, extract_labels, indices, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()
    #print("DEBUG: K-Means",extract_labels.shape)
    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # # k-means clustering
    # kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    # labels = kmeans.labels_
    # print("DRL called")
    nmi, ami, ari = drl_dbscan(X, labels_true)
    # print("DRL call ended")
    # nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    # ami = metrics.adjusted_mutual_info_score(labels_true, labels)
    # ari = metrics.adjusted_rand_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans clustering NMI
    return (n_test_tweets, n_classes, nmi, ami, ari)
