def _easy_worker_init_fn(worker_id : int):
    """
    Sets the seed of the worker to depend on the initial seed. Credit to
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#
# This class will make recording information easier
#
class TrainResult:
    """
    Hold the results of a training round. Basically a nice wrapper.

    Losses and accuracies should be averages over a batch for plot
    labels to make sense.

    Use whichever storage makes sense, but if one doesn't use the
    storage then don't expect the corresponding plots to do anything.
    """

    def __init__(self, model):
        """
        Constructor - prepare local variables. Include model since the
        one provided to trainer will likely be moved to a different
        device. Provide the one that is actually trained here, and then
        it can be used.
        """
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.time_training = 0.0
        self.time_testing = 0.0
        self.model = model

    def plot_loss_train_valid_curves(self, ax, show_legend : bool=True):
        """Use matplotlib to show training curves. Supply axis in ax."""
        ax.plot(self.train_loss_history, label="Train")
        ax.plot(self.test_loss_history, label="Test")
        ax.title(f"Model Loss")
        ax.xlabel("Iteration")
        ax.ylabel("Average Batch Loss")
        if show_legend:
            ax.legend()

    def plot_accuracy_train_valid_curves(self):
        """Use matplotlib to show training curves. Supply axis in ax."""
        ax.plot(self.train_accuracy_history, label="Train")
        ax.plot(self.test_accuracy_history, label="Test")
        ax.title(f"Model Accuracy")
        ax.xlabel("Iteration")
        ax.ylabel("Average Batch Accuracy")
        if show_legend:
            ax.legend()

    def print_time_info(self):
        """Prints some lines with information about timing."""
        print(f"Spent {round(self.time_training)} seconds training.")
        print(f"Spent {round(self.time_testing)} seconds evaluating.")

    def full_analysis(self):
        """Display all analysis plots and print time information."""
        if len(self.train_loss_history) > 0:
            self.plot_loss_train_valid_curves()
        if len(self.train_accuracy_history) > 0:
            self.plot_accuracy_train_valid_curves()
        if self.time_training > 0.0:
            self.print_time_info()

    def save(self, save_filename_func, count : int=0):
        """
        Uses the save_filename_func provided with the count to create
        files that save the models current weights and the training
        progress so far.

        For count, 0 is by convention the final version.
        """
        # Save model
        torch.save(model.state_dict(), save_filename_func(count, True))
        # Save progress
        with open(save_filename_func(count, False), "wb") as file_obj:
            pickle.dump(
                {
                    "train_loss_history" : train_loss_history,
                    "test_loss_history" : test_loss_history,
                    "train_accuracy_history" : train_accuracy_history,
                    "test_accuracy_history" : test_accuracy_history,
                    "time_training" : time_training,
                    "time_testing" : time_testing
                },
                file_obj
            )


def _def_save_filename(iteration_number : int, is_model : bool):
    """
    Returns a save filename from an iteration number, and whether the
    thing being saved is the model or the loss progress.
    """
    # Make sure inputs are okay (no directory traversal attacks!)
    if not isinstance(iteration_number, int):
        raise TypeError("Iteration number should be an integer.")

    # Retrieve proper name
    if is_model:
        os.path.join(
            os.path.dirname(os.path.abspath("")),
            f"model_save_{iteration_number}_{int(time.time())%1000000:06}.pt"
        )
    else:
        os.path.join(
            os.path.dirname(os.path.abspath("")),
            f"progress_save_{iteration_number}_{int(time.time())%1000000:06}.pt"
        )


def _def_get_accuracy(model_output, labels):
    """
    Given model output and labels, finds the average accuracy over the 
    batch.
    """
    preds = model_output.topk(1, dim=1)[1].t().flatten()
    return (pred == labels).sum() / len(pred)

class Trainer:
    """
    Has methods for setting training and testing data, the model, and
    training the model.
    """

    def __init__(self, model):
        """Creates a trainer for a specific model."""
        self._model = model

        self._set_device(torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ))

        # Just a class, will need to freshly call in order to use
        self._optimizer = optim.RMSprop
        # With these parameters after model parameters
        self._optimizer_parameters = {
            "lr" : 1e-4 # Learning rate
        }

        self._loss_function = full_loss

        self._loader_workers = 4
        self._worker_init_fn = _easy_worker_init_fn

        self._save_period = 2
        self._save_filename_func = _def_save_filename

        self._get_accuracy_func = _def_get_accuracy

    def _set_device(self, dev : torch.device):
        """Helper to set device to dev."""
        self._device = dev

        self._model.to(self._device)

    def train(self,
        training_dataset,
        testing_dataset,
        batch_size=256,
        nepochs=4
    ):
        """
        Given a training and testing dataset, how large of batches to 
        extract from them, and the number of epochs to train for, trains
        the models and saves data along the way.
        """
        # Let's get some local variables for brevity
        model = self._model
        optimizer = self._optimizer(
            model.parameters(),
            **self._optimizer_parameters
        )
        loss_func = self._loss_function
        device = self._device

        # Prepare loaders
        def get_loader_from_set(dataset):
            return torch.utils.data.DataLoader(
                dataset,
                batch_size = batch_size,
                num_workers = self._loader_workers,
                worker_init_fn = self._worker_init_fn,
                shuffle = True
            )

        training_loader = get_loader_from_set(training_dataset)
        testing_loader = get_loader_from_set(testing_dataset)

        # Prepare progress bar
        tqdm_variant = tqdm.tqdm # Assume not in Jupyter
        try:
            # Credit https://stackoverflow.com/a/39662359
            shell_name = get_ipython().__class__.__name__
            if shell_name == "ZMQInteractiveShell":
                # Case in Jupyter
                tqdm_variant = tqdm.notebook.tqdm
        except NameError:
            # Probably no iPhython instance, just use standard
            pass

        # Return value
        result = TrainResult(model)
      
        loss = torch.Tensor([0])
        count = 0
        # Credit Steven's notebook for most of this code
        for epoch in tqdm_variant(
            range(nepochs),
            desc=f"Epoch",
            unit="epoch",
            disable=False
        ):
            # Training step
            model.train()
            start = time.perf_counter()
            for (imgs, labels) in tqdm_variant(
                training_loader,
                desc="Training Iteration",
                unit="%",
                disable=False
            ):
                # Prepare optimizer
                optimizer.zero_grad(set_to_none=True)
                
                # Prepare relevant variables
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                
                # Determine loss
                # TODO - broken
                loss = LossFunction(out, labels)

                # Store intermediate results - note average over batch
                result.train_loss_history.append(loss.item())
                result.train_accuracy_history.append(
                    self._get_accuracy_func(out, labels)                 
                )
                
                loss.backward() # Get gradients
                optimizer.step() # Descend gradients
            end = time.perf_counter()
            result.time_training += end - start

            # Testing step
            model.eval()
            start = time.perf_counter()
            with torch.no_grad(): # Don't need gradients
                for (imgs, labels) in tqdm_variant(
                    testing_loader,
                    desc="Testing Iteration",
                    unit="%",
                    disable=False
                ):
                    # Prepare relevant variables
                    imgs, labels = imgs.to(device), labels.to(device)

                    out = model(imgs)
                    
                    # Determine loss
                    loss = LossFunction(out, labels)

                    # Store intermediate results - note average over batch
                    result.test_loss_history.append(loss.item())
                    result.test_accuracy_history.append(
                        self._get_accuracy_func(out, labels)                 
                    )
            end = time.perf_counter()
            result.time_testing += end - start
          
            # Save
            count += 1
            if count % self._save_period == 0:
                result.save(self._save_filename_func, count)

            # Print the last loss calculated and the epoch
            print(f"\nEpoch {epoch}: Loss: {loss.item()}")

        return result

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if isinstance(value, torch.device):
            self._set_device(value)
            return
        elif isinstance(value, str):
            # This may throw an error, in which case purpose achieved
            self._set_device(torch.device(value))
            return
        # Only accept those two types
        raise ValueError("Device must be a torch.device or string.")

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        if not isinstance(value, type):
            raise ValueError("Optimizer must be a class. Perhaps you "\
                "tried to assign an instantiated object?")
        if not hasattr(value, "step"):
            raise ValueError("Optimizer classes must have a step method.")
        if not hasattr(value, "zero_grad"):
            raise ValueError("Optimizer classes must have a zero_grad method.")
        # Presumably safe if we get here
        self._optimizer = value

    @property
    def optimizer_params(self):
        # Danger here: non-string key may be added to parameters
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, value):
        if not isinstance(value, dict):
            raise ValueError("Optimizer parameters should be a "\
                "dictionary of values.")
        # Iterate through keys
        for k in value:
            if not isinstance(k, str):
                raise ValueError("All keys in parameter dictionary must "\
                    "be strings.")
        # Presumably safe if we get here
        self._optimizer_params = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        # Danger here: this only verifies that it is a function
        if callable(value):
            self._loss_function = value
            return
        # If get here, definitely won't work
        raise ValueError("Loss function must be a function.")

    @property
    def loader_workers(self):
        return self._loader_workers

    @loader_workers.setter
    def loader_workers(self, value):
        if isinstance(value, int):
            self._loader_workers = value
            return
        # If get here, definitely won't work
        raise ValueError("Loader worker count must be an integer.")

    @property
    def worker_init_fn(self):
        return self._worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, value):
        # Danger here: this only verifies that it is a function
        if callable(value):
            self._worker_init_fn = value
            return
        # If get here, definitely won't work
        raise ValueError("Worker initialization function must be a function.")

    @property
    def save_period(self):
        return self._save_period

    @save_period.setter
    def save_period(self, value):
        if isinstance(value, int):
            self._save_period = value
            return
        # If get here, definitely won't work
        raise ValueError("Save period must be an integer.")

    @property
    def save_filename_func(self):
        return self._save_filename_func

    @save_filename_func.setter
    def save_filename_func(self, value):
        # Danger here: this only verifies that it is a function
        if callable(value):
            self._save_filename_func = value
            return
        # If get here, definitely won't work
        raise ValueError("Save filename function must be a function.")

    @property
    def get_accuracy_func(self):
        return self._get_accuracy_func

    @get_accuracy_func.setter
    def get_accuracy_func(self, value):
        # Danger here: this only verifies that it is a function
        if callable(value):
            self._get_accuracy_func = value
            return
        # If get here, definitely won't work
        raise ValueError("Get accuracy function must be a function.")

