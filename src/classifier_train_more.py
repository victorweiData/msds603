from metaflow import FlowSpec, step, conda_base, retry, timeout, catch, kubernetes

@conda_base(python='3.9.16',
            libraries={
                'numpy':        '1.23.5',
                'scikit-learn': '1.2.2'
            })
class ClassifierTrainFlow(FlowSpec):

    @kubernetes(image="outerbounds/metaflow:gcp-latest",
                cpu=4, memory=4096, disk=10240)
    @step
    def start(self):
        """
        Load the data, split into train/test, and set up a list of alphas.
        """
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, \
        self.train_labels, self.test_labels = \
            train_test_split(X, y, test_size=0.2, random_state=0)

        print("Data loaded successfully")
        # Try 100 values of alpha between 0.001 and 0.99
        self.lambdas = np.arange(0.001, 1, 0.01)
        self.next(self.train_lasso, foreach='lambdas')

    @kubernetes(image="outerbounds/metaflow:gcp-latest",
                cpu=4, memory=4096, disk=10240)
    @retry(times=3, minutes_between_retries=0.5)
    @timeout(seconds=600)
    @catch(var='error', print_exception=True)
    @step
    def train_lasso(self):
        """
        Train a Lasso model for the current alpha (self.input),
        then pass on to the chooser.
        """
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input, random_state=0)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        """
        Collect all (model, score) pairs, pick the best by test R².
        """
        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(
            (score(inp) for inp in inputs),
            key=lambda tup: -tup[1]
        )
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        """
        Print out all the scores and the winner.
        """
        print("All model scores:")
        for m, s in self.results:
            print(f" • alpha={m.alpha:.3f} → R²={s:.4f}")
        print(f"\nBest model: alpha={self.model.alpha:.3f}")

if __name__ == '__main__':
    ClassifierTrainFlow()