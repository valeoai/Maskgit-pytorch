# Borrowed from https://github.com/nicolas-dufour/diffusion/blob/master/metrics/inception_metrics.py
from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE, _TORCH_FIDELITY_AVAILABLE

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
else:
    class FeatureExtractorInceptionV3(Module):  # type: ignore
        pass

    __doctest_skip__ = ["FrechetInceptionDistance", "FID"]


if _SCIPY_AVAILABLE:
    import scipy

import sklearn.metrics


class NoTrainInceptionV3(FeatureExtractorInceptionV3):
    def __init__(self, name: str, features_list: List[str], feature_extractor_weights_path: Optional[str] = None) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "NoTrainInceptionV3":
        """the inception network should not be able to be switched away from evaluation mode."""
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out
        # return out[0].reshape(x.shape[0], -1)


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class MultiInceptionMetrics(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(self, feature: Union[int, Module] = 2048, reset_real_features: bool = True,
                 compute_unconditional_metrics: bool = False, compute_conditional_metrics: bool = True,
                 compute_conditional_metrics_per_class: bool = True, num_classes: int = 1000,
                 num_inception_chunks: int = 10, manifold_k: int = 3, **kwargs: Any, ) -> None:
        super().__init__(**kwargs)
        self.compute_unconditional_metrics = compute_unconditional_metrics
        self.compute_conditional_metrics = compute_conditional_metrics
        self.compute_conditional_metrics_per_class = compute_conditional_metrics_per_class
        self.num_classes = num_classes
        self.num_inception_chunks = num_inception_chunks
        self.manifold_k = manifold_k
        if isinstance(feature, int):
            num_features = feature
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "FrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = [64, 192, 768, 2048]
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = NoTrainInceptionV3(
                name="inception-v3-compat",
                features_list=[str(feature), "logits_unbiased"],
            ).to("cuda")

        elif isinstance(feature, Module):
            self.inception = feature
            dummy_image = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
            num_features = self.inception(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        self.add_state("real_features", [], dist_reduce_fx=None)

        if self.compute_conditional_metrics:
            self.add_state("fake_cond_features", [], dist_reduce_fx=None)
            self.add_state("fake_cond_logits", [], dist_reduce_fx=None)

        if self.compute_unconditional_metrics:
            self.add_state("fake_uncond_features", [], dist_reduce_fx=None)
            self.add_state("fake_uncond_logits", [], dist_reduce_fx=None)

        if self.compute_conditional_metrics_per_class:
            for i in range(num_classes):
                self.add_state(
                    f"real_cond_features_{i}",
                    [],
                    dist_reduce_fx=None,
                )
                self.add_state(
                    f"fake_cond_features_{i}",
                    [],
                    dist_reduce_fx=None,
                )

                self.add_state(
                    f"fake_cond_logits_{i}",
                    [],
                    dist_reduce_fx=None,
                )

    def update(self, images, labels=None, image_type="unconditional") -> None:  # type: ignore
        if image_type not in ["real", "unconditional", "conditional"]:
            raise ValueError(
                f"Argument `image_type` expected to be one of ['unconditional', 'conditional'], but got {image_type}."
            )
        if image_type == "conditional" and labels is None:
            raise ValueError(
                "Argument `labels` expected to be provided when `image_type` is 'conditional'."
            )
        if image_type == "unconditional" and labels is not None:
            raise ValueError(
                "Argument `labels` expected to be None when `image_type` is 'unconditional'."
            )
        if image_type == "unconditional" and not self.compute_unconditional_metrics:
            raise ValueError(
                "Argument `image_type` is 'unconditional', but `compute_unconditional_metrics` is False."
            )
        if image_type == "conditional" and not self.compute_conditional_metrics:
            raise ValueError(
                "Argument `image_type` is 'conditional', but `compute_conditional_metrics` is False."
            )
        if image_type == "real" and self.compute_conditional_metrics_per_class and labels is None:
            raise ValueError(
                "Argument `labels` expected to be provided when `image_type` is 'real' and `compute_conditional_metrics_per_class` is True."
            )

        features, logits = self.inception(images)

        features = features.view(features.size(0), -1)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
            logits = logits.unsqueeze(0)

        if image_type == "real":
            self.real_features.append(features)
            if self.compute_conditional_metrics_per_class:
                for i in range(self.num_classes):
                    getattr(self, f"real_cond_features_{i}").append(
                        features[labels.argmax(dim=-1) == i]
                    )
        elif image_type == "unconditional":
            self.fake_uncond_features.append(features)
            self.fake_uncond_logits.append(logits)
        elif image_type == "conditional":
            self.fake_cond_features.append(features)
            self.fake_cond_logits.append(logits)
            if self.compute_conditional_metrics_per_class:
                for i in range(self.num_classes):
                    getattr(self, f"fake_cond_features_{i}").append(
                        features[labels.argmax(dim=-1) == i]
                    )
                    getattr(self, f"fake_cond_logits_{i}").append(
                        logits[labels.argmax(dim=-1) == i]
                    )

    def fid(self, real_features, fake_features):
        real_features_mean = real_features.mean(dim=0)
        real_features_cov = self.cov(real_features, real_features_mean)
        fake_features_mean = fake_features.mean(dim=0)
        fake_features_cov = self.cov(fake_features, fake_features_mean)
        return self._compute_fid(
            real_features_mean, real_features_cov, fake_features_mean, fake_features_cov
        ).item()

    def cov(self, features, features_mean):
        features = features - features_mean
        return torch.mm(features.t(), features) / (features.size(0) - 1)

    def _compute_fid(self, mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
        r"""Adjusted version of `Fid Score`_

        The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
        and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

        Args:
            mu1: mean of activations calculated on predicted (x) samples
            sigma1: covariance matrix over activations calculated on predicted (x) samples
            mu2: mean of activations calculated on target (y) samples
            sigma2: covariance matrix over activations calculated on target (y) samples
            eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

        Returns:
            Scalar value of the distance between sets.
        """
        diff = mu1 - mu2

        covmean = sqrtm(sigma1.mm(sigma2))
        # Product might be almost singular
        if not torch.isfinite(covmean).all():
            rank_zero_info(
                f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates"
            )
            offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
            covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

        tr_covmean = torch.trace(covmean)
        return (
            diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
        )

    def inception_score(self, logits):
        idx = torch.randperm(logits.size(0))
        logits = logits[idx]

        prob = logits.softmax(dim=1)
        log_prob = logits.log_softmax(dim=1)

        prob = prob.chunk(self.num_inception_chunks, dim=0)
        log_prob = log_prob.chunk(self.num_inception_chunks, dim=0)
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]

        kl_ = [p * (log_p - torch.log(m_p)) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_).mean()
        return kl.item()

    def compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)
        return dists

    def get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self.compute_pairwise_distance(input_features)
        radii = self.get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def compute_prdc(self, real_features, fake_features, nearest_k):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """

        real_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(real_features, nearest_k)
        fake_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(fake_features, nearest_k)
        distance_real_fake = self.compute_pairwise_distance(real_features, fake_features)

        precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()

        recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

        density = (1.0 / float(nearest_k)) * (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        density = density.sum(axis=0).mean()

        coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

        return precision, recall, density, coverage

    def manifold_metrics(self, real_features, fake_features, nearest_k, num_splits=5):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            nearest_k: int.
            num_splits: int. Number of splits to use for computing metrics.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        real_features = real_features.chunk(num_splits, dim=0)
        fake_features = fake_features.chunk(num_splits, dim=0)
        precision, recall, density, coverage = [], [], [], []
        for real, fake in zip(real_features, fake_features):
            p, r, d, c = self.compute_prdc(real.cpu().numpy(), fake.cpu().numpy(), nearest_k=nearest_k)
            precision.append(torch.tensor(p, device=real.device))
            recall.append(torch.tensor(r, device=real.device))
            density.append(torch.tensor(d, device=real.device))
            coverage.append(torch.tensor(c, device=real.device))
        return (
            torch.stack(precision).mean().item(),
            torch.stack(recall).mean().item(),
            torch.stack(density).mean().item(),
            torch.stack(coverage).mean().item(),
        )

    def compute(self) -> Tensor:
        output_metrics = {}
        real_features = torch.cat(self.real_features, dim=0)
        if self.compute_unconditional_metrics:
            fake_uncond_features = torch.cat(self.fake_uncond_features, dim=0)
            fake_uncond_logits = torch.cat(self.fake_uncond_logits, dim=0)
            output_metrics["fid_unconditional"] = self.fid(
                real_features, fake_uncond_features
            )
            output_metrics["inception_score_unconditional"] = self.inception_score(
                fake_uncond_logits
            )
            (
                output_metrics["precision_unconditional"],
                output_metrics["recall_unconditional"],
                output_metrics["density_unconditional"],
                output_metrics["coverage_unconditional"],
            ) = self.manifold_metrics(
                real_features, fake_uncond_features, self.manifold_k
            )

        if self.compute_conditional_metrics:
            fake_cond_features = torch.cat(self.fake_cond_features, dim=0).to("cuda")
            fake_cond_logits = torch.cat(self.fake_cond_logits, dim=0).to("cuda")
            output_metrics["fid_conditional"] = self.fid(real_features, fake_cond_features)
            output_metrics["inception_score_conditional"] = self.inception_score(fake_cond_logits)
            (
                output_metrics["precision_conditional"],
                output_metrics["recall_conditional"],
                output_metrics["density_conditional"],
                output_metrics["coverage_conditional"],
            ) = self.manifold_metrics(
                real_features, fake_cond_features, self.manifold_k
            )

        if self.compute_conditional_metrics_per_class:
            fid_per_class = 0
            is_per_class = 0
            precision_per_class = 0
            recall_per_class = 0
            density_per_class = 0
            coverage_per_class = 0
            num_classes = 0
            for i in range(self.num_classes):
                if getattr(self, f"real_cond_features_{i}") == []:
                    continue
                num_classes += 1
                if getattr(self, f"real_cond_features_{i}")[0].ndim == 1:
                    real_features_per_class = real_features_per_class.unsqueeze(0)
                real_features_per_class = torch.cat(
                    getattr(self, f"real_cond_features_{i}"), dim=0
                )
                fake_cond_features_per_class = torch.cat(
                    getattr(self, f"fake_cond_features_{i}"), dim=0
                )
                fake_cond_logits_per_class = torch.cat(
                    getattr(self, f"fake_cond_logits_{i}"), dim=0
                )
                fid_per_class += self.fid(
                    real_features_per_class, fake_cond_features_per_class
                )
                is_per_class += self.inception_score(fake_cond_logits_per_class)

                (
                    precision_per_class_i,
                    recall_per_class_i,
                    density_per_class_i,
                    coverage_per_class_i,
                ) = self.manifold_metrics(
                    real_features_per_class,
                    fake_cond_features_per_class,
                    self.manifold_k,
                    num_splits=1,
                )
                precision_per_class += precision_per_class_i
                recall_per_class += recall_per_class_i
                density_per_class += density_per_class_i
                coverage_per_class += coverage_per_class_i

            output_metrics["fid_conditional_per_class"] = fid_per_class / num_classes
            output_metrics["inception_score_conditional_per_class"] = (
                is_per_class / num_classes
            )
            output_metrics["precision_conditional_per_class"] = (
                precision_per_class / num_classes
            )
            output_metrics["recall_conditional_per_class"] = (
                recall_per_class / num_classes
            )
            output_metrics["density_conditional_per_class"] = (
                density_per_class / num_classes
            )
            output_metrics["coverage_conditional_per_class"] = (
                coverage_per_class / num_classes
            )

        return output_metrics

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features = deepcopy(self.real_features)
            if self.compute_conditional_metrics_per_class:
                for i in range(self.num_classes):
                    vars()[f"real_cond_features_{i}"] = deepcopy(
                        getattr(self, f"real_cond_features_{i}")
                    )
            super().reset()
            self.real_features = real_features
            if self.compute_conditional_metrics_per_class:
                for i in range(self.num_classes):
                    setattr(
                        self,
                        f"real_cond_features_{i}",
                        vars()[f"real_cond_features_{i}"],
                    )
        else:
            super().reset()
