# Borrowed from https://github.com/nicolas-dufour/diffusion/blob/master/metrics/sample_and_eval.py
import random
import clip
import torch
from tqdm import tqdm

from Metrics.inception_metrics import MultiInceptionMetrics


def remap_image_torch(image):
    min_norm = image.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    max_norm = image.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    image_torch = ((image - min_norm) / (max_norm - min_norm)) * 255
    image_torch = torch.clip(image_torch, 0, 255).to(torch.uint8)
    return image_torch


class SampleAndEval:
    def __init__(self, device, num_images=50000, compute_per_class_metrics=False, num_classes=1000):
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            reset_real_features=False,
            compute_unconditional_metrics=False,
            compute_conditional_metrics=True,
            compute_conditional_metrics_per_class=compute_per_class_metrics,
            num_classes=num_classes,
            num_inception_chunks=10,
            manifold_k=3,
        )
        self.num_images = num_images
        self.true_features_computed = False
        self.device = device

    def compute_and_log_metrics(self, module):
        with torch.no_grad():
            if not self.true_features_computed or not self.inception_metrics.reset_real_features:
                self.compute_true_images_features(module.test_data)
                self.true_features_computed = True
            self.compute_fake_images_features(module, module.test_data)

            metrics = self.inception_metrics.compute()
            metrics = {f"Eval/{k}": v for k, v in metrics.items()}
            print(metrics)

    def compute_true_images_features(self, dataloader):
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images
        bar = tqdm(dataloader, leave=False, desc="Computing true images features")
        for i, (images, labels) in enumerate(bar):
            if i * dataloader.batch_size >= max_images:
                break

            self.inception_metrics.update(remap_image_torch(images.to(self.device)),
                                          labels.to(self.device),
                                          image_type="real")

    def compute_fake_images_features(self, module, dataloader):
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images

        bar = tqdm(dataloader, leave=False, desc="Computing fake images features")
        for i, (images, labels) in enumerate(bar):
            if i * dataloader.batch_size >= max_images:
                break

            with torch.no_grad():
                if isinstance(labels, list):
                    labels = clip.tokenize(labels[random.randint(0, 4)]).to(self.device)
                    labels = module.clip.encode_text(labels).float()
                else:
                    labels = labels.to(self.device)
                images = module.sample(nb_sample=images.size(0),
                                       labels=labels,
                                       sm_temp=module.args.sm_temp,
                                       w=module.args.cfg_w,
                                       randomize="linear",
                                       r_temp=module.args.r_temp,
                                       sched_mode=module.args.sched_mode,
                                       step=module.args.step)[0]
                images = images.float()
                self.inception_metrics.update(remap_image_torch(images),
                                              labels,
                                              image_type="conditional")
