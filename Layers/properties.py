class ActivationProperties:
    def __init__(self) -> None:
        self.input = None
        self.weights = None
        self.bias = None
        self.output = None
        self.dW = None
        self.db = None
        

    def __getattr__(self, name: str):
        """
        Get Activation Layer Properties for single layer
        """
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name, value):
        """
        Set Activation Layer Properties for single layer
        """
        self.__dict__[f"_{name}"] = value


class ReduceOverfitProperties:
    def __init__(self) -> None:
        self.input = None
        self.weights = None
        self.bias = None
        self.output = None
        self.dW = None
        self.db = None

    def __getattr__(self, name: str):
        """
        Get Dropout/Normalisation Layer Properties for single layer
        """
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name, value):
        """
        Set Dropout/Normalisation Layer Properties for single layer
        """
        self.__dict__[f"_{name}"] = value


class DenseLayerProperties:
    def __init__(self, neurons):
        self.weights = None
        self.bias = None
        self.a = None
        self.z = None
        self.input = None
        self.dW = None
        self.db = None
        self.activation = None
        self.neurons = neurons

    def __getattr__(self, name: str):
        """
        Get Dense Layer Properties for single layer
        """
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name, value):
        """
        Set Dense Layer Properties for single layer
        """
        self.__dict__[f"_{name}"] = value

class ConvLayerProperties:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.channel_out = None
        self.size = None
        self.pad = None
        self.stride = None
        self.filters = None
        
        self.pad_top = None
        self.pad_bottom = None
        self.pad_left = None
        self.pad_right = None
        pass
        

    def __getattr__(self, name: str):
        """
        Get Convolution Layer Properties for single layer
        """
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name, value):
        """
        Set Convolution Layer Properties for single layer
        """
        self.__dict__[f"_{name}"] = value