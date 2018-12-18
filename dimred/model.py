class LinearDimensionalityReduction:
    def __init__(self, method, data, dims):
        self.method = method
        self.data = data
        self.dims = dims
        self._run()

    def _run(self):
        self.method.apply(self)
