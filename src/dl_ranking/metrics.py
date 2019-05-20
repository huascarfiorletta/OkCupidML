import numpy as np


class UserMetrics:

    def __init__(self, prob_data):
        # these dimensions are complementary, need to swap them
        self.indx_p_dominant = prob_data.columns.get_loc("p_dominant")
        self.indx_p_submissive = prob_data.columns.get_loc("p_submissive")

    def user_similarity(self, user_1_indx, user_2_indx, prob_data):
        values_1 = prob_data.iloc[[user_1_indx]].values[0]
        values_2 = prob_data.iloc[[user_2_indx]].values[0]
        values_2[self.indx_p_dominant], values_2[self.indx_p_submissive] = values_2[self.indx_p_submissive], values_2[
            self.indx_p_dominant]

        sum_sq_1 = np.nansum(values_1 * values_1)
        if sum_sq_1 == 0:
            return 0
        sum_sq_2 = np.nansum(values_2 * values_2)
        if sum_sq_2 == 0:
            return 0
        return np.nansum(values_1 * values_2) / (np.sqrt(sum_sq_1) * np.sqrt(sum_sq_2))
