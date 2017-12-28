from seq2seq.models import SimpleSeq2Seq


class Seq2Seq:

    def __init__(self,
                 input_dim=5,
                 hidden_dim=10,
                 output_length=8,
                 output_dim=20,
                 depth=(4, 5),
                 num_epochs=10):

        self.model = SimpleSeq2Seq(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              output_length=output_length,
                              output_dim=output_dim,
                              depth=depth)
        self.model.compile(loss='mse', optimizer='rmsprop')

        self.num_epochs = num_epochs

    def fit(self, data, label):
        self.model.fit(data.reshape(1, data.shape[0], data.shape[1]), label, self.num_epochs, verbose=2)

    def predict(self, data):

        yhat = self.model.predict(data.reshape(1, data.shape[0], data.shape[1]))
        return yhat[0, 0]


