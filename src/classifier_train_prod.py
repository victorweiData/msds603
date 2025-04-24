from metaflow import FlowSpec, step, conda_base, resources, kubernetes, retry, timeout, catch
import numpy as np

@conda_base(
    python='3.9.16',
    libraries={
        'numpy': '1.23.5',
        'scikit-learn': '1.2.2'
    }
)
class ClassifierTrainFlow(FlowSpec):

    @resources(cpu=2, memory=8192)
    @kubernetes(image="outerbounds/metaflow:gcp-latest")
    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        # Load data
        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = \
            train_test_split(X, y, test_size=0.2, random_state=0)
        print("Data loaded successfully")

        # set of alpha values to try
        self.lambdas = np.arange(0.001, 1, 0.01)
        self.next(self.train_lasso, foreach='lambdas')

    @retry(times=3, minutes_between_retries=0.5)
    @timeout(seconds=600)
    @catch(var='error', print_exception=True)
    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print("Scores:")
        for m, s in self.results:
            print(f"{m.__class__.__name__} {s:.4f}")
        print("Best model:", self.model)

if __name__ == '__main__':
    ClassifierTrainFlow()