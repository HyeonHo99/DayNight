import torch
import torchvision.utils

from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from torch.utils.tensorboard import SummaryWriter

def train_fn(disc_night, disc_day, gen_day, gen_night, loader, opt_disc, opt_gen,l1, mse,d_scaler, g_scaler):
    night_reals = 0
    night_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (day, night) in enumerate(loop):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators night and day
        with torch.cuda.amp.autocast():
            fake_night = gen_night(day)
            D_night_real = disc_night(night)
            D_night_fake = disc_night(fake_night.detach())
            night_reals += D_night_real.mean().item()
            night_fakes += D_night_fake.mean().item()
            D_night_real_loss = mse(D_night_real, torch.ones_like(D_night_real))
            D_night_fake_loss = mse(D_night_fake, torch.zeros_like(D_night_fake))
            D_night_loss = D_night_real_loss + D_night_fake_loss

            fake_day = gen_day(night)
            D_day_real = disc_day(day)
            D_day_fake = disc_day(fake_day.detach())
            D_day_real_loss = mse(D_day_real, torch.ones_like(D_day_real))
            D_day_fake_loss = mse(D_day_fake, torch.zeros_like(D_day_fake))
            D_day_loss = D_day_real_loss + D_day_fake_loss

            # put it together
            D_loss = (D_night_loss + D_day_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators night and day
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_night_fake = disc_night(fake_night)
            D_day_fake = disc_day(fake_day)
            loss_G_night = mse(D_night_fake, torch.ones_like(D_night_fake))
            loss_G_day = mse(D_day_fake, torch.ones_like(D_day_fake))

            # cycle loss
            cycle_day = gen_day(fake_night)
            cycle_night = gen_night(fake_day)
            cycle_day_loss = l1(day, cycle_day)
            cycle_night_loss = l1(night, cycle_night)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_day = gen_day(day)
            identity_night = gen_night(night)
            identity_day_loss = l1(day, identity_day)
            identity_night_loss = l1(night, identity_night)

            # add all together
            G_loss = (
                loss_G_day
                + loss_G_night
                + cycle_day_loss * config.LAMBDA_CYCLE
                + cycle_night_loss * config.LAMBDA_CYCLE
                + identity_night_loss * config.LAMBDA_IDENTITY
                + identity_day_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()



        if config.SAVE_IMAGES and idx % 20 == 0:
            save_image(fake_night*0.5+0.5, f"saved_images/0509/day2night_{idx}.png")
            save_image(fake_day*0.5+0.5, f"saved_images/0509/night2day_{idx}.png")

        loop.set_postfix(night_real=night_reals/(idx+1), night_fake=night_fakes/(idx+1))



def main():
    ## tensorboard
    writer_fake_day = SummaryWriter(f"runs/0509/night2day")
    writer_fake_night = SummaryWriter(f"runs/0509/day2night")
    step = 1

    disc_night = Discriminator(in_channels=3).to(config.DEVICE)
    disc_day = Discriminator(in_channels=3).to(config.DEVICE)
    gen_day = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_night = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_night.parameters()) + list(disc_day.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_day.parameters()) + list(gen_night.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_night, gen_night, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_day, gen_day, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_night, disc_night, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_day, disc_day, opt_disc, config.LEARNING_RATE,
        )

    dataset = DayNightDataset(
        root_day=config.TRAIN_DIR+"/day", root_night=config.TRAIN_DIR+"/night", transform=config.transforms
    )
    val_dataset = DayNightDataset(
       root_day=config.VAL_DIR+"/day", root_night=config.VAL_DIR+"/night", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=24,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_night, disc_day, gen_day, gen_night, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        ## save on tensorboard
        with torch.no_grad():
            for day,night in val_loader:
                day = day.to(config.DEVICE)
                night = night.to(config.DEVICE)

                fake_night = gen_night(day).reshape(-1,3,config.IMG_SIZE,config.IMG_SIZE)
                fake_day = gen_day(night).reshape(-1,3,config.IMG_SIZE,config.IMG_SIZE)

                img_grid_fake_day = torchvision.utils.make_grid(fake_day,normalize=True)
                img_grid_fake_night = torchvision.utils.make_grid(fake_night,normalize=True)

                writer_fake_day.add_image("night2day", img_grid_fake_day, global_step=step)
                writer_fake_night.add_image("day2night", img_grid_fake_night, global_step=step)
                break
        step += 1

        if config.SAVE_MODEL and (epoch+1)%10 == 0:
            save_checkpoint(gen_night, opt_gen, filename=config.CHECKPOINT_GEN_night)
            save_checkpoint(gen_day, opt_gen, filename=config.CHECKPOINT_GEN_day)
            save_checkpoint(disc_night, opt_disc, filename=config.CHECKPOINT_CRITIC_night)
            save_checkpoint(disc_day, opt_disc, filename=config.CHECKPOINT_CRITIC_day)

if __name__ == "__main__":
    main()