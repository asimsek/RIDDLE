import numpy as np
import torch
from .integrity import require_finite


# Keep every finite event when applying a preprocessing transform learned on a
# separate reference sample.  Values outside the fitted min/max range are
# saturated only for the numerically singular logit step instead of being
# deleted.  This is especially important for anomaly searches, where tail
# events can be the signal-like population of interest.
LOGIT_EPS = 1.0e-6


class LHCORD_data_handler:
    def __init__(
        self,
        inner_train_path,
        inner_test_path,
        outer_train_path,
        outer_test_path,
        inner_extrasig_path,
        inner_extrabkg_path=None,
        inner_val_path=None,
        batch_size=256,
        test_batch_size=None,
        device=torch.device("cpu"),
    ):
        self.datashift = 0.0
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = 10 * self.batch_size
        else:
            self.test_batch_size = test_batch_size
        self.device = device
        self.innerdata_train = load_files(inner_train_path)
        self.innerdata_test = load_files(inner_test_path)
        self.outerdata_train = load_files(outer_train_path)
        self.outerdata_test = load_files(outer_test_path)
        if inner_val_path is not None:
            self.innerdata_val = load_files(inner_val_path)
        else:
            self.innerdata_val = None
        if inner_extrasig_path is not None:
            self.innerdata_extrasig = load_files(inner_extrasig_path)
        else:
            self.innerdata_extrasig = None
        if inner_extrabkg_path is not None:
            self.innerdata_extrabkg = load_files(inner_extrabkg_path)
        else:
            self.innerdata_extrabkg = None
        self.original_innerdata_train = self.innerdata_train
        self.original_innerdata_test = self.innerdata_test
        self.original_outerdata_train = self.outerdata_train
        self.original_outerdata_test = self.outerdata_test
        self.original_innerdata_val = self.innerdata_val
        self.original_innerdata_extrasig = self.innerdata_extrasig
        self.original_innerdata_extrabkg = self.innerdata_extrabkg
        self.fiducial_cut = None
        self.outer_ANODE_datadict_train = None
        self.inner_ANODE_datadict_train = None
        self.outer_ANODE_datadict_test = None
        self.inner_ANODE_datadict_test = None
        self.inner_ANODE_datadict_val = None
        self.inner_ANODE_datadict_extrasig = None
        self.inner_ANODE_datadict_extrabkg = None
        self.outer_ANODE_inner_preprocessing_datadict_train = None
        self.outer_ANODE_inner_preprocessing_datadict_test = None

    def preprocess_ANODE_data(
        self, fiducial_cut=False, no_logit=False, no_mean_shift=False, external_param=None
    ):
        self.fiducial_cut = fiducial_cut
        self.outer_ANODE_datadict_train = load_dataset(
            self.outerdata_train,
            batch_size=self.batch_size,
            fiducial_cut=fiducial_cut,
            shuffle_loader=True,
            no_logit=no_logit,
            device=self.device,
            no_mean_shift=no_mean_shift,
            external_datadict=external_param,
        )
        self.outer_ANODE_datadict_test = load_dataset(
            self.outerdata_test,
            batch_size=self.test_batch_size,
            external_datadict=self.outer_ANODE_datadict_train if external_param is None else external_param,
            fiducial_cut=fiducial_cut,
            shuffle_loader=False,
            no_logit=no_logit,
            device=self.device,
            no_mean_shift=no_mean_shift,
        )
        self.inner_ANODE_datadict_train = load_dataset(
            self.innerdata_train,
            batch_size=self.batch_size,
            fiducial_cut=fiducial_cut,
            shuffle_loader=True,
            no_logit=no_logit,
            device=self.device,
            no_mean_shift=no_mean_shift,
            external_datadict=self.outer_ANODE_datadict_train,
        )
        self.inner_ANODE_datadict_test = load_dataset(
            self.innerdata_test,
            batch_size=self.test_batch_size,
            external_datadict=self.outer_ANODE_datadict_train,
            fiducial_cut=fiducial_cut,
            shuffle_loader=False,
            no_logit=no_logit,
            device=self.device,
            no_mean_shift=no_mean_shift,
        )
        if self.innerdata_val is not None:
            self.inner_ANODE_datadict_val = load_dataset(
                self.innerdata_val,
                batch_size=self.test_batch_size,
                external_datadict=self.outer_ANODE_datadict_train,
                fiducial_cut=fiducial_cut,
                shuffle_loader=False,
                no_logit=no_logit,
                device=self.device,
                no_mean_shift=no_mean_shift,
            )
        if self.innerdata_extrasig is not None:
            self.inner_ANODE_datadict_extrasig = load_dataset(
                self.innerdata_extrasig,
                batch_size=self.test_batch_size,
                external_datadict=self.outer_ANODE_datadict_train,
                fiducial_cut=fiducial_cut,
                shuffle_loader=False,
                no_logit=no_logit,
                device=self.device,
                no_mean_shift=no_mean_shift,
            )
        if self.innerdata_extrabkg is not None:
            self.inner_ANODE_datadict_extrabkg = load_dataset(
                self.innerdata_extrabkg,
                batch_size=self.test_batch_size,
                external_datadict=self.outer_ANODE_datadict_train,
                fiducial_cut=fiducial_cut,
                shuffle_loader=False,
                no_logit=no_logit,
                device=self.device,
                no_mean_shift=no_mean_shift,
            )


def load_files(file_path):
    if isinstance(file_path, list):
        file_list = []
        for path in file_path:
            file_list.append(np.load(path).astype("float32"))
        loaded_file = np.vstack(file_list)
    else:
        loaded_file = np.load(file_path).astype("float32")
    return loaded_file


def _clip_unit_interval(data, eps=LOGIT_EPS):
    return torch.clamp(data, min=float(eps), max=1.0-float(eps))


def logit_transform(data, datamax, datamin, domain_cut=False, fiducial_cut=False, tail_safe=False):
    data2 = (data - datamin) / (datamax - datamin)
    if fiducial_cut:
        mask = torch.prod((data2 > 0.05) & (data2 < 0.95), 1).type(torch.bool)
        data3 = data2[mask]
    elif domain_cut and not tail_safe:
        # Preserve the historical reference-fit behavior.  The result-affecting
        # problem was applying these fitted limits as an acceptance cut to
        # independent validation/SR/test events.
        mask = torch.prod((data2 > 0) & (data2 < 1), 1).type(torch.bool)
        data3 = data2[mask]
    else:
        # Once preprocessing is frozen on a separate reference sample,
        # production RIDDLE uses a non-rejecting tail-safe transform: all finite
        # rows survive and only the unit-interval coordinate is saturated before
        # the logit.  This avoids preferentially dropping anomalous tail events.
        mask = torch.ones(data2.shape[0], dtype=torch.bool, device=data2.device)
        data3 = _clip_unit_interval(data2)
    data4 = torch.log(data3 / (1 - data3))
    require_finite(data4, "Logit-transformed mapped features")
    return (data4, mask)


def logit_transform_inverse(data, datamax, datamin):
    dataout = (datamin + datamax * np.exp(data)) / (1 + np.exp(data))
    return dataout


def quick_logit(x):
    x_norm = (x - min(x)) / (max(x) - min(x))
    x_norm = x_norm[(x_norm != 0) & (x_norm != 1)]
    logit = np.log(x_norm / (1 - x_norm))
    logit = logit[~np.isnan(logit)]
    return logit


def create_dataset(data, device=torch.device("cpu")):
    sigorbg = torch.from_numpy(data[:, -1]).to(device)
    labels = torch.from_numpy(data[:, 0:1]).to(device)
    tensor = torch.from_numpy(data[:, 1:-1]).to(device)
    return (sigorbg, labels, tensor)


def load_dataset(
    data,
    batch_size=256,
    external_datadict=None,
    fiducial_cut=False,
    shuffle_loader=False,
    cond_range=None,
    no_logit=False,
    device=torch.device("cpu"),
    no_mean_shift=False,
):
    require_finite(data, "Physical input preprocessing")
    if cond_range is not None:
        assert len(cond_range) == 2, "cond_range must be 2-element array-like object!"
        cond_mask = np.logical_and(data[:, 0] > cond_range[0], data[:, 0] < cond_range[1])
        input_data = data[cond_mask]
    else:
        input_data = data
    datadict = {}
    datadict["sigmask"] = torch.from_numpy(input_data[:, -1] == 1).to(device)
    datadict["bgmask"] = torch.from_numpy(input_data[:, -1] == 0).to(device)
    datadict["sigorbg"], datadict["labels"], datadict["tensor"] = create_dataset(input_data, device=device)
    if external_datadict is not None:
        datadict["max"] = external_datadict["max"].clone()
        datadict["min"] = external_datadict["min"].clone()
    else:
        datadict["max"] = torch.max(datadict["tensor"], dim=0).values
        datadict["min"] = torch.min(datadict["tensor"], dim=0).values
    if torch.any(datadict["max"] <= datadict["min"]):
        raise ValueError("Preprocessing requires nonconstant feature domains")
    if no_logit:
        if fiducial_cut:
            _, mask = logit_transform(
                datadict["tensor"],
                datadict["max"],
                datadict["min"],
                domain_cut=True,
                fiducial_cut=fiducial_cut,
            )
            tensor2 = datadict["tensor"][mask]
        else:
            tensor2 = (datadict["tensor"] - datadict["min"]) / (datadict["max"] - datadict["min"])
            if external_datadict is None:
                mask = ((tensor2 > 0.0) & (tensor2 < 1.0)).all(axis=1)
                tensor2 = tensor2[mask]
            else:
                mask = torch.ones(tensor2.shape[0], dtype=torch.bool, device=tensor2.device)
                tensor2 = _clip_unit_interval(tensor2)
    else:
        tensor2, mask = logit_transform(
            datadict["tensor"], datadict["max"], datadict["min"], domain_cut=True,
            fiducial_cut=fiducial_cut, tail_safe=external_datadict is not None
        )
    if external_datadict is not None:
        datadict["mean2"] = external_datadict["mean2"].clone()
        datadict["std2"] = external_datadict["std2"].clone()
        datadict["std2_logit_fix"] = external_datadict["std2_logit_fix"].clone()
    else:
        if fiducial_cut:
            reference_tensor_logit_fix, _ = logit_transform(
                datadict["tensor"], datadict["max"], datadict["min"], domain_cut=True, fiducial_cut=False
            )
            if no_logit:
                reference_tensor = datadict["tensor"]
            else:
                reference_tensor = reference_tensor_logit_fix
        else:
            reference_tensor_logit_fix, _ = logit_transform(
                datadict["tensor"],
                datadict["max"],
                datadict["min"],
                domain_cut=True,
                fiducial_cut=fiducial_cut,
            )
            reference_tensor = tensor2
        datadict["mean2"] = torch.mean(reference_tensor, dim=0)
        datadict["std2"] = torch.std(reference_tensor, dim=0)
        datadict["std2_logit_fix"] = torch.std(reference_tensor_logit_fix, dim=0)
    datadict["sigmask"] = datadict["sigmask"][mask]
    datadict["bgmask"] = datadict["bgmask"][mask]
    datadict["sigorbg"] = datadict["sigorbg"][mask]
    datadict["labels"] = datadict["labels"][mask]
    datadict["tensor"] = datadict["tensor"][mask]
    datadict["mask"] = mask
    if not no_mean_shift:
        datadict["tensor2"] = (tensor2 - datadict["mean2"]) / datadict["std2"]
    else:
        datadict["tensor2"] = tensor2
    require_finite(datadict["tensor2"], "Normalized mapped features")
    datadict["dataset"] = torch.utils.data.TensorDataset(datadict["tensor2"], datadict["labels"])
    datadict["loader"] = torch.utils.data.DataLoader(
        datadict["dataset"], batch_size=batch_size, shuffle=shuffle_loader
    )
    return datadict


def stack_data(data_array, cond_labels, sig_labels=None, samples=False):
    if samples:
        data_or_sample = np.zeros((data_array.shape[0], 1))
        sig_labels = 5 * np.ones((data_array.shape[0], 1))
    else:
        assert sig_labels is not None, "Valid signal or background labels need to be provided!"
        data_or_sample = np.ones((data_array.shape[0], 1))
    stacked_data = np.hstack(
        (cond_labels.reshape((-1, 1)), data_array, data_or_sample, sig_labels.reshape((-1, 1)))
    ).astype("float32")
    from .integrity import require_finite
    require_finite(stacked_data, "Development latent stacking")
    return stacked_data
