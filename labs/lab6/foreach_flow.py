from metaflow import FlowSpec, step

class ForeachFlow(FlowSpec):
    @step
    def start(self):
        self.creatures = ['bird', 'mouse', 'dog']
        self.next(self.analyze, foreach='creatures')

    @step
    def analyze(self):
        print("Analyzing", self.input)
        self.creature = self.input
        self.score    = len(self.input)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.best = max(inputs, key=lambda t: t.score).creature
        self.next(self.end)

    @step
    def end(self):
        print(self.best, "won!")
        
if __name__ == '__main__':
    ForeachFlow()
