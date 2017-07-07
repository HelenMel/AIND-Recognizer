import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        max_BIC = None; max_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=self.verbose).fit(self.X, self.lengths)
                n_features = hmm_model.n_features
                logL = hmm_model.score(self.X, self.lengths)
                p = num_states * num_states + 2 * num_states * n_features - 1
                N = len(self.X)
                BIC = -2 * logL + p * math.log(N)
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))
                    print("score {0:.2f}".format(BIC))
                if max_BIC is None or max_BIC > BIC:
                    max_BIC = BIC; max_model = hmm_model
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
    
        return max_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_DIC = None; max_model = None
        
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=self.verbose).fit(self.X, self.lengths)
        
                # find scores for other words
                other_words_scores = []
                for (word, (X, length)) in self.hwords.items():
                    if word == self.this_word:
                        continue
                    other_words_scores.append(hmm_model.score(X, length))
            
                other_words_scores_average = sum(other_words_scores)/len(other_words_scores)
                
                # find difference between this word and all other words
                this_word_score = hmm_model.score(self.X, self.lengths)
                DIC = this_word_score - other_words_scores_average
                if max_DIC is None or max_DIC < DIC:
                    max_DIC = DIC; max_model = hmm_model
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
        
        return max_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        max_CV_score = None; max_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=self.verbose)
            scores = []
            
            split_n = len(self.sequences) if len(self.sequences) < 3 else 3
            if split_n == 1:
                try:
                    scores.append(hmm_model.fit(self.X, self.lengths).score(self.X, self.lengths))
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))
                        print("score {0:.2f}".format(s))
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
            else:
                split_method = KFold(n_splits=split_n, random_state=self.random_state)
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        train_x, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        test_x, test_length = combine_sequences(cv_test_idx, self.sequences)
                        scores.append(hmm_model.fit(train_x, train_lengths).score(test_x, test_length))
                        if self.verbose:
                            print("model created for {} with {} states".format(self.this_word, num_states))
                            print("score {0:.2f}".format(s))
                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(self.this_word, num_states))
            if len(scores) > 0:
                CV_score = np.array(scores).mean()
                if max_CV_score is None or max_CV_score < CV_score:
                    max_CV_score = CV_score; max_model = hmm_model
    
        return max_model

