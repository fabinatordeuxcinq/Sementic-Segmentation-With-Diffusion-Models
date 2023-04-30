# Sementic-Segmentation-With-Diffusion-Models
Forked project from Julia Wolleb project : https://github.com/JuliaWolleb/Diffusion-based-Segmentation

This is a improved version that allow Semantic Segmentation and more general framework.

## Usage 
Training a diffusion model for Semantic Segmentation can be done by calling,
__segmentation_train.py__ file. 
To do it on your data, you need to define a Dataset object (see torch.data.Dataset or the section below)
and edit __segmentation_train.py__.
A conviniant way of making calls to __segmentation_train.py__ is to use a .sh script. 
Example is show in train.sh.

Then, once you have a trained model, (that should be place in *./result/*_nb_training_step.pt*)
you can use __segmentation_sample.py__ to generate semgmentaiton masks with this model.
Again, you will need to edit that file to define your own Dataset and to edit sample.sh. 

During sampling, you can visualize the denoising process with tensorboard bu passing the __--tb_display__
parameter. Sampled exemples will be save as well in local (only finals samples) 
You can use __--save_numpy__ to save raw model outputs (multi channels image). 
To have a better visualization of the segmentation masks, you can pass __--save_nice__ to save 
a nice verion of masks with colors for each classes.

## Defining your own Dataset 

The procedur follow the torch way of doing it 

```

class MyDataset(torch.utils.data.Dataset) : 

  def __init__ (self, ..., test_flag)
    self.test_flag = test_flag
    ...
  def __len__(self) : 
    ...
  def __getitem__(self, idx) : 
    ...
    if self.test_flag : 
      return image, image_indice_or_name
    else : 
      return image, label 
```

## TO DO 
Pillow import si redondent w/ scikitimage, ditch one of them
Add data, results, nince vizaulisations

