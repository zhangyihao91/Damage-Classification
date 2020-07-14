class LabelSmoothing(nn.Module):

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion=nn.KLDivLoss(reduction='sum')
        self.confidence=1.0-smoothing
        self.smoothing=smoothing
        self.size=size
        self.true_dist=None
        
    def forward(self, x, target):
        assert x.size(1)== self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist=true_dist
        
        return self.criterion(x, true_dist)
