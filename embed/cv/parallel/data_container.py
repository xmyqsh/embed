from embed.cv.parallel import DataContainer

class DataContainerWithPad(DataContainer):
    def __init__(self,
                 data,
                 stack=False,
                 pad=False,
                 **kwargs):
        assert not (stack and pad), (stack, pad)
        super(DataContainerWithPad, self).__init__(data, stack=stack, **kwargs)
        self._pad = pad

    @property
    def pad(self):
        return self._pad
