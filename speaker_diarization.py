import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as at
from pyAudioAnalysis import MidTermFeatures as mtf
from scipy.spatial import distance
import sklearn.cluster

def speaker_diarization(filename, n_speakers, mid_window=2.0, mid_step=0.2,
                        short_window=0.05, lda_dim=35, plot_res=False):

    # Read audio file and preprocess
    sampling_rate, signal = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(signal) / sampling_rate

    # Load pre-trained speaker classification models
    base_dir = 'models'
    classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _ = at.load_model_knn(os.path.join(base_dir, "knn_speaker_10"))
    classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _,  _ = at.load_model_knn(os.path.join(base_dir, "knn_speaker_male_female"))

    # Extract mid-term and short-term features
    mid_feats, st_feats, _ = mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * short_window),
                                   round(sampling_rate * short_window * 0.5))

    # Initialize a feature matrix for speaker diarization
    mid_term_features = np.zeros((mid_feats.shape[0] + len(class_names_all) + len(class_names_fm), mid_feats.shape[1]))

    # Speaker classification and feature aggregation
    for index in range(mid_feats.shape[1]):
        feature_norm_all = (mid_feats[:, index] - mean_all) / std_all
        feature_norm_fm = (mid_feats[:, index] - mean_fm) / std_fm
        _, p1 = at.classifier_wrapper(classifier_all, "knn", feature_norm_all)
        _, p2 = at.classifier_wrapper(classifier_fm, "knn", feature_norm_fm)
        start = mid_feats.shape[0]
        end = mid_feats.shape[0] + len(class_names_all)
        mid_term_features[0:mid_feats.shape[0], index] = mid_feats[:, index]
        mid_term_features[start:end, index] = p1 + 1e-4
        mid_term_features[end::, index] = p2 + 1e-4

    mid_feats = mid_term_features

    # Feature selection
    feature_selected = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    mid_feats = mid_feats[feature_selected, :]

    # Normalize features
    mid_feats_norm, mean, std = at.normalize_features([mid_feats.T])
    mid_feats_norm = mid_feats_norm[0].T
    n_wins = mid_feats.shape[1]

    # Remove outliers
    dist_all = np.sum(distance.squareform(distance.pdist(mid_feats_norm.T)), axis=0)
    m_dist_all = np mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.2 * m_dist_all)[0]

    mt_feats_norm_or = mid_feats_norm
    mid_feats_norm = mid_feats_norm[:, i_non_outliers]

    # LDA dimensionality reduction
    if lda_dim > 0:
        window_ratio = int(round(mid_window / short_window))
        step_ratio = int(round(short_window / short_window))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        for index in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])

        for index in range(num_of_features):
            cur_pos = 0
            feat_len = len(st_feats[index])
            while cur_pos < feat_len:
                n1 = cur_pos
                n2 = cur_pos + window_ratio
                if n2 > feat_len:
                    n2 = feat_len
                short_features = st_feats[index][n1:n2]
                mt_feats_to_red[index].append(np.mean(short_features))
                mt_feats_to_red[index + num_of_features].append(np.std(short_features))
                cur_pos += step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] + len(class_names_all) + len(class_names_fm), mt_feats_to_red.shape[1]))
        limit = mt_feats_to_red.shape[0] + len(class_names_all)
        for index in range(mt_feats_to_red.shape[1]):
            feature_norm_all = (mt_feats_to_red[:, index] - mean_all) / std_all
            feature_norm_fm = (mt_feats_to_red[:, index] - mean_fm) / std_fm
            _, p1 = at.classifier_wrapper(classifier_all, "knn", feature_norm_all)
            _, p2 = at.classifier_wrapper(classifier_fm, "knn", feature_norm_fm)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0], index] = mt_feats_to_red[:, index]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]:limit, index] = p1 + 1e-4
            mt_feats_to_red_2[limit::, index] = p2 + 1e-4
        mt_feats_to_red = mt_feats_to_red_2
        mt_feats_to_red = mt_feats_to_red[feature_selected, :]
        mt_feats_to_red, mean, std = at.normalize_features([mt_feats_to_red.T])
        mt_feats_to_red = mt_feats_to_red[0].T
        labels = np.zeros((mt_feats_to_red.shape[1], ))
        lda_step = 1.0
        lda_step_ratio = lda_step / short_window
        for index in range(labels.shape[0):
            labels[index] = int(index * short_window / lda_step_ratio)

    # Clustering to determine the number of speakers
    if n_speakers <= 0:
        s_range = range(2, 5)
    else:
        s_range = [n_speakers]
    cluster_labels = []
    sil_all = []
    cluster_centers = []

    for speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=speakers)
        k_means.fit(mid_feats_norm.T)
        cls = k_means.labels_
        means = k_means.cluster_centers_

        cluster_labels.append(cls)
        cluster_centers.append(means)
        sil_1 = []
        sil_2 = []
        for c in range(speakers):
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clust_per_cent < 0.020
