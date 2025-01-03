class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        """ Save activations from the target layer """
        activation = output
        if isinstance(activation, tuple):
            # Handle case where output is a tuple (e.g., return value from a layer)
            activation = activation[0]  # Select the first element if it's a tuple

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        """ Save gradients from the target layer """
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # Gradients are only stored if output requires gradients
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        """ Perform a forward pass and extract activations and gradients """
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        """ Remove registered hooks """
        for handle in self.handles:
            handle.remove()
