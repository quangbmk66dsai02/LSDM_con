from posa.dataset import ProxDataset_txt, HUMANISE
from torch.utils.data import DataLoader, Dataset
from model.sdm import SceneDiffusionModel

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


train_data_dir = "data/protext/proxd_train"
max_frame = 256
fix_ori = True
jump_step = 8

train_dataset = ProxDataset_txt(train_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                    step_multiplier=1, jump_step=jump_step)

print(type(train_dataset[0]))
print(len(train_dataset[0]))
print("obj mask:   ", train_dataset[0][0])
print("obj cat:    \n",train_dataset[0][2])

print(train_dataset[0][4])

 #        return obj_mask, obj_verts, obj_cats, target_verts, target_cat, text_prompt

def load_data(
    *,
    dataset,
    batch_size,
    image_size = 256,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
        )
    while True:
        yield from loader
    
data = load_data(dataset=train_dataset, batch_size= 2)

