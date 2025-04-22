from metaflow import FlowSpec, step, Parameter

class Counterflow(FlowSpec):

    # Define a CLI parameter named --ct (default=20, int, required)
    begin_count = Parameter(
        'ct',
        help='Initial value for the counter',
        default=20,
        type=int,
        required=True
    )

    @step
    def start(self):
        # initialize using the parameter
        self.count = self.begin_count
        print(f"Starting count = {self.count}")
        self.next(self.add)

    @step
    def add(self):
        print("The count is", self.count, "before incrementing")
        self.count += 1
        self.next(self.end)

    @step
    def end(self):
        self.count += 1
        print("Final count is", self.count)

if __name__ == '__main__':
    Counterflow()
