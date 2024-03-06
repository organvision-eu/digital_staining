import torch
from torchvision.transforms import ToTensor
import lightning as L
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import io

from ..utils.losses import GenLoss
from ..utils.visualize import display_batch, hstack_figures
from ..utils.metrics import classification_metrics


class WGANGP(L.LightningModule):

    def __init__(self, Gen, Disc, config):
        super().__init__()
        self.automatic_optimization = False
        self.L1Loss = torch.nn.L1Loss(reduction="none")

        self.target_channels = config['target_channels']
        self.len_val_loader = config['len_val_loader']
        self.use_classification_metric = config['use_classification_metric']
        self.adversarial_training = config['adversarial_training']

        if self.use_classification_metric:  # compute classification metric
            self.classifier = config['classifier']
            self.classification_loader = config['classification_loader']
            self.classifier.freeze()

        # networks
        self.generator = Gen
        self.gen_criterion = GenLoss(loss='l1')

        if self.adversarial_training:
            self.discriminator = Disc

        # val step variables
        self.fig_list = []
        self.counter = 0

        # hyperparameters
        self.lr_g = config['lr_g']  # generator
        self.lr_d = config['lr_d']  # discriminator (critic)

    def forward(self, z):
        return self.generator(z)

    def compute_gp(self, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)

        # Interpolation between real data and fake data.
        interpolation = (eps * real_data + (1 - eps) *
                         fake_data).requires_grad_(True)

        # get logits for interpolated images
        interp_logits = self.discriminator(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

    def training_step_wgan(self, batch, batch_idx):
        n_critic = 5  # number of training steps for discriminator per iter
        opt_g, opt_d = self.optimizers()

        x, y_real = batch

        mask = ~(y_real.isnan())

        y_real = torch.nan_to_num(y_real, nan=0.0)

        batch_size = x.shape[0]

        y_fake = self.generator(x)

        y_fake_masked = y_fake*mask

        ##########################
        # Optimize Discriminator #
        ##########################
        real_concat_with_input = torch.cat((x, y_real), 1)
        real_out = self.discriminator(real_concat_with_input).mean()

        fake_concat_with_input = torch.cat((x, y_fake_masked.detach()), 1)
        fake_out = self.discriminator(fake_concat_with_input).mean()

        lambda_gp = 10
        gp = self.compute_gp(real_concat_with_input, fake_concat_with_input)

        was_loss = fake_out - real_out + lambda_gp * gp
        was_loss.create_graph = True  # enable backprop for gradient penalty

        opt_d.zero_grad()
        self.manual_backward(was_loss, retain_graph=True)
        opt_d.step()

        ######################
        # Optimize Generator #
        ######################
        if batch_idx % n_critic == 0:  # update generator every n_critic steps
            gen_fake_concat_with_input = torch.cat((x, y_fake_masked), 1)
            disc_fake_out = self.discriminator(gen_fake_concat_with_input)
            adv_weight = 0.05
            epoch = self.current_epoch
            g_loss, l1_loss_train, adv_loss = self.gen_criterion(
                disc_fake_out, y_fake_masked, y_real, epoch=epoch, mask=mask, adv_weight=adv_weight)  # compute generator loss

            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()

            self.log_dict({"g_loss": g_loss, "was_loss": was_loss,
                           "gp": gp, "adv_loss": adv_loss,
                           "train_L1": l1_loss_train,
                           }, prog_bar=True)

    def training_step_unet(self, batch, batch_idx):  # without adv loss
        opt_g = self.optimizers()

        x, y_real = batch

        mask = ~(y_real.isnan())

        y_real = torch.nan_to_num(y_real, nan=0.0)

        y_fake = self.generator(x)

        y_fake_masked = y_fake*mask

        # Optimize Generator #
        l1_loss_train = self.L1Loss(y_fake_masked, y_real)[mask].mean()
        g_loss = l1_loss_train
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        if batch_idx % 5 == 0:
            self.log_dict({"g_loss": g_loss,
                           "train_L1": l1_loss_train,
                           }, prog_bar=True)

    def training_step(self, batch, batch_idx):
        if self.adversarial_training:
            self.training_step_wgan(batch, batch_idx)
        else:
            self.training_step_unet(batch, batch_idx)

    def configure_optimizers(self):
        opt_g = torch.optim.RMSprop(
            params=self.generator.parameters(), lr=self.lr_g)
        if self.adversarial_training:
            opt_d = torch.optim.RMSprop(
                params=self.discriminator.parameters(), lr=self.lr_d)
            return opt_g, opt_d

        return opt_g

    def on_train_epoch_start(self):
        torch.cuda.synchronize()

    def on_validation_epoch_end(self):
        if not self.use_classification_metric:
            return
        f1_macro, f1_micro, f1_weighted = 0, 0, 0
        for batch in self.classification_loader:
            f1_macro_batch, f1_micro_batch, f1_weighted_batch = classification_metrics(
                self.generator, self.classifier, batch)
            f1_macro += f1_macro_batch
            f1_micro += f1_micro_batch
            f1_weighted += f1_weighted_batch
        n_samples = len(self.classification_loader)
        f1_macro /= n_samples
        f1_micro /= n_samples
        f1_weighted /= n_samples
        self.log("train_f1_macro", f1_macro, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("train_f1_micro", f1_micro, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("train_f1_weighted", f1_weighted,
                 on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_real = batch
        with torch.no_grad():
            y_fake = self.generator(x)
        mask = ~(y_real.isnan())

        l1loss = self.L1Loss(y_fake, y_real)
        l1loss_mean = l1loss[mask].mean()
        self.log("val_L1", l1loss_mean, on_step=False,
                 on_epoch=True, prog_bar=True)

        if self.use_classification_metric:
            f1_macro, f1_micro, f1_weighted = classification_metrics(
                self.generator, self.classifier, batch, prediction=y_fake)
            self.log("f1_macro", f1_macro, on_step=False,
                     on_epoch=True, prog_bar=True)
            self.log("f1_micro", f1_micro, on_step=False,
                     on_epoch=True, prog_bar=True)
            self.log("f1_weighted", f1_weighted, on_step=False,
                     on_epoch=True, prog_bar=True)

        for i, ch in enumerate(self.target_channels):
            channel_loss = l1loss[:, i, ...][mask[:, i, ...]].mean()
            if not channel_loss.isnan():
                self.log(f"L1_{ch}", channel_loss)

        if self.counter > 5:
            tensorboard = self.logger.experiment
            # stack horizontally the figures in self.fig_list
            image = hstack_figures(self.fig_list)
            tensorboard.add_image('val_fig', ToTensor()(
                image), global_step=self.global_step)
            self.fig_list = []
            self.counter = 0

        elif batch_idx % (self.len_val_loader//6) == 0:
            fig = display_batch(x, y_real, pred=y_fake, target_channel_names=self.target_channels,
                                limit_images=16, show3d='middle', batch=self.counter)
            canvas = FigureCanvasAgg(fig)
            # Get the renderer's buffer and string it into a PNG
            buf = io.BytesIO()
            canvas.print_png(buf)
            img = Image.open(buf)
            self.fig_list.append(img)
            self.counter += 1

        return l1loss
