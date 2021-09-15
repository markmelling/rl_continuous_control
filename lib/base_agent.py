
import torch

class BaseAgent:

    def save(self, filename='checkpoint'):
        torch.save(self.local_network.state_dict(), '%s.pth' % (filename))

    def load(self, filename='checkpoint'):
        # state_dict = torch.load('%s.pth' % filename)
        state_dict = torch.load('%s.pth' % filename, map_location=lambda storage, loc: storage)
        # print('state keys', state_dict.keys())
        self.local_network.load_state_dict(state_dict)
        self.local_network.eval()
        self.target_network.load_state_dict(self.local_network.state_dict())

