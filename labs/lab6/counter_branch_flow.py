from metaflow import FlowSpec, step

class CounterBranchFlow(FlowSpec):
    @step
    def start(self):
        self.creature = "dog"
        self.count    = 0
        self.next(self.add_one, self.add_two)

    @step
    def add_one(self):
        self.count += 1
        self.next(self.join)

    @step
    def add_two(self):
        self.count += 2
        self.next(self.join)

    @step
    def join(self, inputs):
        # 1) pick the branch value you want
        self.count = max(inp.count for inp in inputs)
        # 2) merge the rest of the artifacts (creature, etc.)
        self.merge_artifacts(inputs)
        print("counts were:", inputs.add_one.count, "and", inputs.add_two.count)
        self.next(self.end)

    @step
    def end(self):
        print(f"The creature is {self.creature} and final count = {self.count}")

if __name__ == '__main__':
    CounterBranchFlow()
