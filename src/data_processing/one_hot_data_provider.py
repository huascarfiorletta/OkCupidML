import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing.constants import likes
from dl_ranking.metrics import UserMetrics


class OneHotDataProvider:

    def __init__(self):
        # load questions csv
        questions_data = pd.read_csv('/Users/huascar/workspace/Jupyter/OkCupidData/question_data.csv', header=0, delimiter=';')
        questions_data = questions_data.rename(columns={questions_data.columns[0]: "q"})
        # keep only question lines
        questions_data = questions_data[questions_data.q.str[0] == 'q']
        # sort them by non empty number of lines
        questions_data.sort_values(by=['N'], ascending=False)
        orig_data = pd.read_pickle("/Users/huascar/workspace/Jupyter/OkCupidData/unprocessed.bz2")

        ###split data in questions, demographics, probabilities###
        self.prob_data = orig_data[[col for col in orig_data.columns if col.startswith('p')]]
        # self.question_data = orig_data[[col for col in orig_data.columns if col.startswith('q')]]
        demog_data_columns = [col for col in orig_data.columns if not col.startswith('q') and not col.startswith('p')]
        self.demog_data = orig_data[demog_data_columns]
        # self.demog_question_data = orig_data[[col for col in orig_data.columns if not col.startswith('p')]]
        self.UserMetrics = UserMetrics(self.prob_data)
        self.one_hot = pd.read_pickle("/Users/huascar/workspace/Jupyter/OkCupidData/one_hot.bz2")

    def get_training_data(self, samples):
        X = [[],[]]
        Y = []
        with tqdm(total=samples) as pbar:
            while len(Y) < samples:
                user_1_indx, user_2_indx = self.sample_compatible_pair()
                similarity = self.UserMetrics.user_similarity(user_1_indx, user_2_indx, self.prob_data)
                if similarity == 0:
                    continue
                user_1_data = self.one_hot.iloc[[user_1_indx]].values[0]
                user_2_data = self.one_hot.iloc[[user_2_indx]].values[0]
                X[0].append(user_1_data)
                X[1].append(user_2_data)
                Y.append(similarity)
                pbar.update(1)
        return [np.array(X[0]),np.array(X[1])], np.array(Y)

    def sample_compatible_pair(self):
        while True:
            indx_1 = np.random.randint(len(self.demog_data))
            user_1 = self.demog_data.iloc[[indx_1]]
            user_1_orientation = user_1['gender_orientation'].values[0]
            if user_1_orientation not in likes.keys():
                continue
            liked_orientations = likes[user_1_orientation]
            while True:
                indx_2 = np.random.randint(len(self.demog_data))
                if indx_1 == indx_2:
                    continue
                user_2 = self.demog_data.iloc[[indx_2]]
                user_2_orientation = user_2['gender_orientation'].values[0]
                if user_2_orientation in liked_orientations:
                    #print user_1_orientation, user_2_orientation
                    return indx_1, indx_2
